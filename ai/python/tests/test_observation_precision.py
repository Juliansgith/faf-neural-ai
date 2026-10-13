"""Cache/live numeric parity, including values not representable in float16."""
import unittest
import numpy as np
from fa_env.observation_precision import PRECISION_CONTRACT
from fa_train import tools_bc as bc
from fa_train.worker import compact
from fa_train.tools_trainer import widen
from fa_train.tools_model import empty_labels

def observation(n,nc,ne):
    rng=np.random.default_rng(n+nc+ne)
    real=lambda shape:rng.random(shape,dtype=np.float32)
    return {'tokens':real((n,70)),'token_bp':np.arange(n,dtype=np.int64),
            'token_order':np.zeros(n,np.int64),'token_type':np.zeros(n,np.int64),
            'token_last':np.zeros(n,np.int64),'token_last_feat':real((n,3)),
            'mask_anchor':np.ones(n,bool),'mask_silo':np.ones((n,2),bool),'mask_toggle':np.ones((n,8),bool),
            'mask_subject':np.ones((len(bc.V_INDEX),n),bool),'mask_target':np.ones((5,n),bool),
            'cand_bp':np.arange(nc,dtype=np.int64),'cand_kind':np.zeros(nc,np.int64),'cand_feat':real((nc,34)),
            'enh_idx':np.arange(ne,dtype=np.int64),'enh_feat':real((ne,11)),
            'can_build':np.zeros((0,2),np.int64),'can_enh':np.zeros((0,2),np.int64),
            'map':real((16,32,32)),'scalars':real((34+len(bc.V_INDEX),)),
            'privileged':real((20,)),'belief':real((10,)),'mask_verb':np.ones(len(bc.V_INDEX),bool),
            'mask_build_cell':np.ones((4,32,32),bool),'mask_move_cell':np.ones((32,32),bool),'fine':np.array([1,1,256,256],np.float32),'focus':np.zeros((3,3),np.float32)}

class ObservationPrecisionTest(unittest.TestCase):
    def test_cache_roundtrip_matches_real_actor_transport_for_ragged_and_empty_inputs(self):
        rows=[observation(3,2,4),observation(1,0,0),observation(7,3,2)]
        packed=bc.pack(rows,[empty_labels(len(x['tokens'])) for x in rows]);games=object.__new__(bc.Games)
        self.assertEqual(packed['cand_feat'].dtype,np.float32);self.assertEqual(packed['enh_feat'].dtype,np.float32)
        for i,raw in enumerate(rows):
            actual=widen(compact(raw));cached=games.step_obs(packed,i)
            for key in cached:np.testing.assert_array_equal(cached[key],actual[key],err_msg=key)

    def test_transport_preserves_previous_numeric_behavior_and_does_not_mutate_input(self):
        x=observation(3,2,1);original={k:v.copy() for k,v in x.items()};wire=compact(x);actor=widen(wire)
        for key in x:
            np.testing.assert_array_equal(x[key],original[key])
            expected=x[key].astype(np.float16).astype(np.float32) if key in ('tokens','map','token_last_feat') else x[key]
            np.testing.assert_array_equal(actor[key],expected)
        self.assertIs(wire['cand_feat'],x['cand_feat']);self.assertIs(wire['enh_feat'],x['enh_feat'])

    def test_legacy_candidate_rounding_would_fail_parity(self):
        x=observation(3,2,1)
        for key in ['cand_feat','enh_feat']:
            self.assertFalse(np.array_equal(x[key],x[key].astype(np.float16).astype(np.float32)))

    def test_cache_requires_current_format_and_explicit_precision_contract(self):
        current={'format':bc.CACHE_FORMAT,'observation_precision':PRECISION_CONTRACT,'encoder_sha256':'source','key':'key'}
        self.assertTrue(bc._cache_is_current(current,'source','key'))
        self.assertFalse(bc._cache_is_current({**current,'format':bc.CACHE_FORMAT - 1},'source','key'))
        self.assertFalse(bc._cache_is_current({**current,'observation_precision':'old-rounding'},'source','key'))
        missing=dict(current);missing.pop('observation_precision')
        self.assertFalse(bc._cache_is_current(missing,'source','key'))

if __name__=='__main__':unittest.main()
