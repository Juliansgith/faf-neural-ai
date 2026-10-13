"""Represent explicit cancellation of an observed build with an existing STOP slot."""


def add_build_cancellations(commands, actions, units, building_orders, queue_verbs):
    cleared = {str(uid) for a in actions if a.get('type') == 'clear' for uid in a.get('units', [])}
    seen_queue = set()
    result = []
    for command in commands:
        subject = command.get('subject') or []
        if (command['verb'] == 'build' and len(subject) == 1 and subject[0] in cleared
                and subject[0] not in seen_queue
                and units.get(subject[0], {}).get('order') in building_orders):
            # Pair metadata exists only during label filtering. It is not a new policy head.
            pair = len(result)
            result.append({'verb': 'stop', 'subject': list(subject), '_build_cancel_pair': pair})
            command = {**command, '_build_cancel_pair': pair}
        result.append(command)
        if command['verb'] in queue_verbs:
            seen_queue.update(subject)
    return result


def atomic_cancellation_groups(accepted):
    """Drop broken pairs; never let filtering leave a destructive orphan STOP."""
    groups, rejected = [], []
    index = 0
    while index < len(accepted):
        command = accepted[index]
        pair = command.get('_build_cancel_pair')
        if pair is None:
            groups.append([command])
        elif (command['verb'] == 'stop' and index + 1 < len(accepted)
              and accepted[index + 1].get('_build_cancel_pair') == pair
              and accepted[index + 1]['verb'] == 'build'
              and accepted[index + 1]['subject'] == command['subject']):
            groups.append([command, accepted[index + 1]])
            index += 1
        else:
            rejected.append(command)
        index += 1
    return groups, rejected
