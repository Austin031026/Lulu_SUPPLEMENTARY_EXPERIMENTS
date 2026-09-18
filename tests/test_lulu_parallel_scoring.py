"""ReN's independent teacher/recognition services join before on-policy updates."""
from __future__ import annotations

import copy

import pytest

from lulu import persistent
from test_lulu_persistent import _args, _PipelineHarness


class _OrderedServices(_PipelineHarness):
    """Exhaust one service before allowing the other to complete anything."""
    def __init__(self, fastest, method='ren_opd'):
        super().__init__(method=method)
        self.fastest = fastest
        self.updates = []
        self.requests = []

    def sent(self, conn, message):
        if message['op'] == 'score':
            self.requests.append((conn.role, copy.deepcopy(message)))
            if conn.role == 'hindsight':
                # Teacher-first completion must not discard an unsubmitted H prompt.
                assert all(record['hindsight_prompt_ids'] for record in message['records'])
            else:
                assert all('hindsight_prompt_ids' not in record and 'recognition_ids' not in record
                           for record in message['records'])
        if message['op'] == 'update':
            # Neither a fast Teacher nor a fast privileged model can release an
            # update before the entire other service has finished this snapshot.
            assert sum(e[:3] == ('receive', 'hindsight', 'scored') for e in self.events) == 2
            assert sum(e[:3] == ('receive', 'teacher', 'scored') for e in self.events) == 2
            self.updates.append(copy.deepcopy(message))
        return super().sent(conn, message)

    def receive(self, connections):
        slowest = 'hindsight' if self.fastest == 'teacher' else 'teacher'
        for role in (self.fastest, 'student', slowest):
            for index, (conn, message) in enumerate(self.pending):
                if conn in connections and conn.role == role:
                    self.pending.pop(index)
                    self.events.append(('receive', conn.role, message['op'], conn.rank))
                    return conn, message
        raise AssertionError('Pipeline waited for unavailable service output')


@pytest.mark.parametrize('fastest', ['teacher', 'hindsight'])
def test_ren_forks_both_services_and_joins_all_results_regardless_of_finish_order(fastest):
    workers = _OrderedServices(fastest)
    result = persistent.pipeline_round(_args(round=3, global_batch_prompts=4), workers,
                                      workers.students, workers.hindsight, workers.teacher)
    first_h_send = workers.events.index(('send', 'hindsight', 'score', None))
    first_t_send = workers.events.index(('send', 'teacher', 'score', None))
    first_h_result = workers.events.index(('receive', 'hindsight', 'scored', None))
    first_t_result = workers.events.index(('receive', 'teacher', 'scored', None))
    assert max(first_h_send, first_t_send) < min(first_h_result, first_t_result)
    slowest = 'hindsight' if fastest == 'teacher' else 'teacher'
    last_fast_result = max(i for i, e in enumerate(workers.events)
                           if e[:3] == ('receive', fastest, 'scored'))
    first_slow_result = workers.events.index(('receive', slowest, 'scored', None))
    assert last_fast_result < first_slow_result
    assert len(workers.updates) == 2
    for update in workers.updates:
        for target in update['records'].values():
            assert target['recognition_ids'].shape == (2, 2)
            assert target['teacher_hidden'].shape == (2, 16)
            assert target['teacher_scored']
    assert result['hindsight_seconds'] == pytest.approx(.25)
    assert result['teacher_seconds'] == pytest.approx(.25)
    assert not workers.pending


@pytest.mark.parametrize('method', ['ren_graft', 'union_topk'])
def test_sparse_objectives_keep_hindsight_before_teacher_dependency(method):
    workers = _OrderedServices('teacher', method=method)
    persistent.pipeline_round(_args(method=method, round=3, global_batch_prompts=4), workers,
                              workers.students, workers.hindsight, workers.teacher)
    first_h_result = workers.events.index(('receive', 'hindsight', 'scored', None))
    first_t_send = workers.events.index(('send', 'teacher', 'score', None))
    assert first_h_result < first_t_send
    for role, message in workers.requests:
        if role == 'teacher':
            assert all('correction_ids' in record for record in message['records'])
