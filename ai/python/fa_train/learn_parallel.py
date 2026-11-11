"""learn_rl's first pass and PPO epochs on worker processes (--processes, CPU only; Linux: fork).

Why processes: on the server (2026-11-07) one thread team of 28 was no faster than 14, and 16 worker threads reached
only about 8 busy cores: much of a decision's work is Python (assembling chunks, the decoder's many small tensors),
which threads cannot share.

Why no rounds: 28 processes waiting for each round's slowest stream kept only 7 cores busy (a late-game chunk costs
many times an early one). Streams do not depend on each other, only on their own memory, so each free worker takes the
next chunk of a stream that is not being worked on, and the optimizer steps after every --chunks-per-step finished
chunks. The first pass needs no streams at all: a worker takes a whole game. A gradient can be a few steps old when
it is applied; at this learning rate a step moves very little.

The parameters and the recurrent memory live in shared memory, so a worker always sees the latest step. The master
never starts OpenMP before forking (it runs with one thread), so each worker can start its own team. Data the epochs
read (values, advantages and so on) is inherited read-only by copy-on-write: the pool is forked after they are
computed.
"""
from __future__ import annotations

import mmap
import multiprocessing as mp
from multiprocessing.connection import wait

import numpy as np
import torch

_FORK = mp.get_context("fork")


def _shared(shape: tuple, dtype=np.float32) -> np.ndarray:
    n = int(np.prod(shape)) * np.dtype(dtype).itemsize
    return np.frombuffer(mmap.mmap(-1, max(n, 1)), dtype=dtype, count=int(np.prod(shape))).reshape(shape)



def share_network(net) -> None:
    for p in net.parameters():
        p.data.share_memory_()
    for b in net.buffers():
        b.share_memory_()


def _pool(worker, count: int, args: tuple) -> tuple[list, list]:
    conns, procs = [], []
    for index in range(count):
        mine, theirs = _FORK.Pipe()
        proc = _FORK.Process(target=worker, args=(theirs, index, *args), daemon=True)
        proc.start()
        theirs.close()
        conns.append(mine)
        procs.append(proc)
    return conns, procs


def _dispatch(conns: list, jobs: list, should_stop=None) -> list:
    """Every job to the next free worker; the results in any order."""
    results, free, busy = [], list(conns), set()
    jobs = list(jobs)
    while jobs or busy:
        if should_stop is not None and should_stop():
            jobs = []
        while jobs and free:
            c = free.pop()
            c.send(jobs.pop(0))
            busy.add(c)
        for c in wait(list(busy)):
            try:
                results.append(c.recv())
            except EOFError as exc:
                raise RuntimeError("a learner worker died") from exc
            busy.discard(c)
            free.append(c)
    return results


def _stop(conns: list, procs: list) -> None:
    for c in conns:
        try:
            c.send(None)
        except (BrokenPipeError, OSError):
            pass
    for proc in procs:
        proc.join(timeout=30)
        if proc.is_alive():
            proc.terminate()


# ------------------------------------------------------------------------------------------------- first pass
def _first_pass_worker(conn, index, net, ref, games, lengths, a, threads, out, chunk_batch, micro_view):
    torch.set_num_threads(threads)
    values, log_rho, kl0, prox, refp, features = out
    L = a.length
    device = torch.device("cpu")
    group = torch.tensor([0])
    with torch.no_grad():
        for gi in iter(conn.recv, None):
            state = net.initial_state(1, device)
            ref_state = ref.initial_state(1, device)
            kl_sum, kl_n = 0.0, 0.0
            off = games.games[gi]["offsets_t"]
            for start in range(0, lengths[gi], L):
                c = chunk_batch(games, [(gi, start)], L, device)
                n = min(L, lengths[gi] - start)
                sub, b, act, _ = micro_view(c, group, 1, L)
                h, m, z, state = net.core(b, state, L, sub["starts"], return_state=True)
                v = net.value_of(z, b).float()
                _, ent, _ = net.score(h, m, z, b, act, temperature=a.temperature)
                ulp = ent["unit_logp"].float()
                known = ent["unit_known"].float()
                if refp is not None:
                    hr, mr, zr, ref_state = ref.core(b, ref_state, L, sub["starts"], return_state=True)
                    _, ent_r, _ = ref.score(hr, mr, zr, b, act, temperature=a.temperature)
                    ulp_ref = ent_r["unit_logp"].float().numpy()
                ulp_np = ulp.numpy()
                for t in range(n):
                    o, e = int(off[start + t]), int(off[start + t + 1])
                    prox[gi][o:e] = ulp_np[t, : e - o]
                    if refp is not None:
                        refp[gi][o:e] = ulp_ref[t, : e - o]
                values[gi][start:start + n] = v[:n].numpy()
                if features is not None:
                    features[gi][0][start:start + n] = z[:n].float().numpy()
                    features[gi][1][start:start + n] = b["privileged"][:n].float().numpy()
                if a.vtrace:
                    d_unit = (ulp - sub["old_ulp"]) * known
                    per_decision = d_unit.sum(1) / known.sum(1).clamp(min=1.0)
                    ok = c["valid"] > 0
                    kl_unit = (((d_unit.exp() - 1) - d_unit) * known).sum(1) / known.sum(1).clamp(min=1.0)
                    kl_sum += float((kl_unit * ok).sum())
                    kl_n += float(ok.sum())
                    has = np.asarray(games.games[gi]["pol_has"][start:start + n], np.float32)
                    log_rho[gi][start:start + n] = per_decision[:n].numpy() * has
                    kl0[gi][start:start + n] = kl_unit[:n].numpy()
            conn.send((kl_sum, kl_n))


def first_pass(net, ref, games, usable, lengths, tokens_of, a, processes, with_ref, with_features, chunk_batch, micro_view,
               threads):
    """The values, ratios and reference scores of every decision, as learn_rl's serial first pass computes them."""
    values = {i: _shared((lengths[i],)) for i in usable}
    log_rho = {i: _shared((lengths[i],)) for i in usable}
    kl0 = {i: _shared((lengths[i],)) for i in usable}
    prox = {i: _shared((tokens_of[i],)) for i in usable}
    refp = {i: _shared((tokens_of[i],)) for i in usable} if with_ref else None
    features = None
    if with_features:
        priv = int(np.asarray(games.games[usable[0]]["privileged"][0]).shape[0])
        features = {i: (_shared((lengths[i], net.core_dim)), _shared((lengths[i], priv))) for i in usable}
    share_network(net)
    share_network(ref)
    out = (values, log_rho, kl0, prox, refp, features)
    conns, procs = _pool(_first_pass_worker, processes,
                         (net, ref, games, lengths, a, threads, out, chunk_batch, micro_view))
    kl_sum, kl_n = 0.0, 0.0
    try:
        jobs = sorted(usable, key=lambda i: -tokens_of[i])
        for x, y in _dispatch(conns, jobs, lambda: a.stop_file and a.stop_file.exists()):
            kl_sum += x
            kl_n += y
    finally:
        _stop(conns, procs)
    if a.stop_file and a.stop_file.exists():
        raise SystemExit(f"{a.stop_file} found: stopping without saving")
    return values, log_rho, kl0, prox, refp, features, kl_sum, kl_n


# ------------------------------------------------------------------------------------------------------ epochs
def _epoch_worker(conn, index, net, games, a, threads, memory, grad_buffer, data, chunk_batch, ppo_group):
    torch.set_num_threads(threads)
    H, C = memory
    params = [p for p in net.parameters() if p.requires_grad]
    L = a.length
    device = torch.device("cpu")
    flat = grad_buffer[index]
    group = torch.tensor([0])
    for job in iter(conn.recv, None):
        s, item, valid_scale, sup_scale = job
        c = chunk_batch(games, [item], L, device, data["values"], data["adv"], data["ret"], data["adv_up"], data["loc"],
                        data["kl0"], data["prox"], data["refp"])
        new_h, new_c = torch.zeros(1, 1, net.core_dim), torch.zeros(1, 1, net.core_dim)
        grads, stats = ppo_group(net, c, group, (H[:, [s]].contiguous(), C[:, [s]].contiguous()), new_h, new_c, 1, a, params,
                                 valid_scale, sup_scale)
        H[:, s], C[:, s] = new_h[:, 0], new_c[:, 0]
        o = 0
        for p, g in zip(params, grads):
            n = p.numel()
            if g is None:
                flat[o:o + n] = 0.0
            else:
                # reshape, not view: a convolution's gradient comes back channels_last when its input is (the map
                # network on the CPU learner); reshape flattens in the parameter's own (logical) order either way
                flat[o:o + n] = g.reshape(-1)
            o += n
        # the master reads this buffer before it gives this worker anything else
        conn.send((index, [(k, float(v)) for k, v in stats]))


class EpochPool:
    """Workers forked after the advantages are known. run_epoch(streams, opt) is one pass over the games."""

    def __init__(self, net, games, usable, lengths, a, processes, threads, chunk_batch, ppo_group, data,
                 kl_stop: bool = True):
        self.kl_stop = kl_stop
        from fa_env.factory_upgrades import label_prefix
        share_network(net)
        self.net, self.games, self.a, self.lengths = net, games, a, lengths
        self.params = [p for p in net.parameters() if p.requires_grad]
        self.count = sum(p.numel() for p in self.params)
        self.memory = tuple(torch.zeros(1, a.streams, net.core_dim).share_memory_() for _ in range(2))
        self.grads = [torch.zeros(self.count).share_memory_() for _ in range(processes)]
        self.per_step = int(a.chunks_per_step or a.streams)
        # a step's denominators: what a full round of this many chunks holds on average (rounds no longer exist)
        prefix = label_prefix(games)
        decisions = valid = sup = 0
        for gi in usable:
            g = games.games[gi]
            n = lengths[gi]
            decisions += n
            valid += int(np.asarray(g["pol_has"][:n]).astype(bool).sum())
            if getattr(games, "teacher", {}).get(gi, True):
                sup += int(np.asarray(g[prefix + "unit_supervise"][:n]).astype(bool).sum())
        size = self.per_step * a.length
        self.valid_scale = max(1.0, size * valid / max(decisions, 1))
        self.sup_scale = max(1.0, size * sup / max(decisions, 1))
        self.conns, self.procs = _pool(_epoch_worker, processes,
                                       (net, games, a, threads, self.memory, self.grads, data, chunk_batch, ppo_group))

    def _apply(self, total, opt) -> float:
        opt.zero_grad(set_to_none=True)
        o = 0
        for p in self.params:
            n = p.numel()
            p.grad = total[o:o + n].view_as(p).clone()
            o += n
        a = self.a
        norm = float(torch.nn.utils.clip_grad_norm_(self.net.parameters(), a.max_grad))
        a.step = getattr(a, "step", 0) + 1
        for group in opt.param_groups:
            group["lr"] = group.get("base_lr", a.lr) * min(1.0, a.step / max(1, a.warmup))
        finite = np.isfinite(norm)
        if finite:
            opt.step()
        opt.zero_grad(set_to_none=True)
        return norm if finite else 0.0

    def run_epoch(self, queues: list, opt) -> tuple[dict, bool]:
        """Every stream's games, chunk by chunk. Returns the summed statistics (as learn_rl's epoch keeps them, one
        "n" per optimizer step) and whether the KL stop ended the epoch."""
        a, L, lengths = self.a, self.a.length, self.lengths
        for x in self.memory:
            x.zero_()
        position = [[0, 0] for _ in queues]                      # [index in queue, start in game]
        done = [0] * len(queues)
        busy_stream: dict = {}
        free = list(self.conns)
        by_index = {index: c for index, c in enumerate(self.conns)}
        total = torch.zeros(self.count)
        in_total = 0
        sums = {k: 0.0 for k in ("pg", "v", "ent", "kl", "clip", "ref_kl", "distill", "local", "kl_data", "gnorm", "n")}
        stopped = False

        def next_stream():
            best = None
            for s, q in enumerate(queues):
                if s in busy_stream.values() or position[s][0] >= len(q):
                    continue
                if best is None or done[s] < done[best]:
                    best = s
            return best

        while True:
            while free and not stopped:
                s = next_stream()
                if s is None:
                    break
                qi, start = position[s]
                gi = queues[s][qi]
                c = free.pop()
                c.send((s, (gi, start), self.valid_scale, self.sup_scale))
                busy_stream[c] = s
                start += L
                position[s] = [qi + 1, 0] if start >= lengths[gi] else [qi, start]
                done[s] += 1
            if not busy_stream:
                break
            if a.stop_file and a.stop_file.exists():
                self.close()
                raise SystemExit(f"{a.stop_file} found: stopping without saving")
            for c in wait(list(busy_stream), timeout=5.0):
                try:
                    index, stats = c.recv()
                except EOFError as exc:
                    raise RuntimeError("a learner worker died") from exc
                total += self.grads[index]
                in_total += 1
                for k, v in stats:
                    sums[k] = sums.get(k, 0.0) + v
                del busy_stream[c]
                free.append(by_index[index])
                if in_total >= self.per_step:
                    sums["gnorm"] += self._apply(total, opt)
                    sums["n"] += 1.0
                    total.zero_()
                    in_total = 0
                    if self.kl_stop and sums["n"] >= 8 and sums.get("kl", 0.0) / sums["n"] > 1.5 * a.target_kl:
                        stopped = True
        if in_total >= self.per_step / 2 and not stopped:
            sums["gnorm"] += self._apply(total, opt)
            sums["n"] += 1.0
        return sums, stopped

    def close(self) -> None:
        _stop(self.conns, self.procs)
