"""Typed candidate meaning before features, then complete governed handoff.

This additive adapter leaves the scientific codec, features and trained state
unchanged. New descriptions require fresh features. Actions/interpretations
remain proposals; rendering receipts are not execution or acoustic evidence.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import math
import re

from src.alice_personality.identity.codec import canonical, fingerprint
from src.alice_personality.identity.consumer import RenderedResponse
from src.alice_personality.identity.contracts import IdentityError
from scripts.eipm.gemma_n0 import cumulative_successor, governed_response


def _require(condition, message):
    if not condition:
        raise IdentityError(message)


def _text(value):
    _require(isinstance(value, str) and bool(value.strip()), 'typed candidate needs complete text')


def _json_data(value):
    if type(value) is dict:
        _require(all(type(key) is str and key for key in value), 'typed payload keys must be strings')
        for item in value.values():
            _json_data(item)
    elif type(value) is list:
        for item in value:
            _json_data(item)
    else:
        _require(value is None or type(value) in (str, bool, int)
                 or type(value) is float and math.isfinite(value), 'typed payload must be finite JSON data')


@dataclass(frozen=True)
class TypedCandidate:
    candidate_id: str
    kind: str
    payload: dict

    def validate(self):
        _text(self.candidate_id)
        fields = {'utterance': {'text', 'delivery'}, 'interpretation': {'proposition'},
                  'action': {'operation', 'description', 'arguments'}}
        _require(isinstance(self.kind, str) and self.kind in fields, 'unknown typed candidate kind')
        _require(type(self.payload) is dict and set(self.payload) == fields[self.kind],
                 'typed candidate payload fields differ from its kind')
        _json_data(self.payload)
        if self.kind == 'utterance':
            text, delivery = self.payload['text'], self.payload['delivery']
            _text(text)
            _require(delivery is None or type(delivery) is dict, 'typed delivery is explicit or unavailable')
            if delivery is not None:
                _require('intent' in delivery and set(delivery) <= {'intent', 'emphasis_spans', 'pause_locations'},
                         'typed delivery needs qualitative intent and explicit text locations')
                _text(delivery['intent'])
                for key in ('emphasis_spans', 'pause_locations'):
                    _require(key not in delivery or type(delivery[key]) is list, 'typed speech locations must be ordered lists')
                for span in delivery.get('emphasis_spans', []):
                    _require(type(span) is dict and set(span) == {'start', 'end', 'text', 'intent'},
                             'typed emphasis needs exact speech-text spans')
                    _require(type(span['start']) is int and type(span['end']) is int
                             and 0 <= span['start'] < span['end'] <= len(text)
                             and text[span['start']:span['end']] == span['text'], 'typed emphasis span differs from speech text')
                    _text(span['intent'])
                for pause in delivery.get('pause_locations', []):
                    _require(type(pause) is dict and set(pause) == {'offset', 'intent'}, 'typed pause needs a speech-text position')
                    _require(type(pause['offset']) is int and 0 <= pause['offset'] <= len(text), 'typed pause position is outside speech text')
                    _text(pause['intent'])
        elif self.kind == 'interpretation':
            _text(self.payload['proposition'])
        else:
            _text(self.payload['operation'])
            _text(self.payload['description'])
            _require(type(self.payload['arguments']) is dict, 'typed action arguments must be an explicit object')

    def semantic_text(self):
        self.validate()
        # Meaning only: candidate IDs, authority flags and teacher targets are
        # not semantic features. The entire executable/proposed content is.
        return 'Candidate meaning (proposal, not an observed outcome):\n' + canonical(
            {'kind': self.kind, 'payload': self.payload,
             'text_position_unit': 'unicode_code_point'}).decode('utf-8')


@dataclass(frozen=True)
class PreparedCandidates:
    document: object
    candidates: tuple[TypedCandidate, ...]
    binding: dict
    binding_sha256: str

    def validate(self):
        self.document.validate()
        _require(type(self.candidates) is tuple and all(type(c) is TypedCandidate for c in self.candidates),
                 'typed candidates require their complete ordered mapping')
        _require([c.candidate_id for c in self.candidates] == [c.candidate_id for c in self.document.candidates],
                 'typed candidates differ from the ordered document')
        for candidate, entry in zip(self.candidates, self.document.candidates):
            _require(candidate.semantic_text() == entry.text, 'typed payload differs from produced candidate meaning')
        _require(type(self.binding) is dict and set(self.binding) ==
                 {'schema', 'original_document_sha256', 'compiled_document_sha256', 'candidates_sha256'},
                 'typed candidate binding is incomplete')
        _require(self.binding['schema'] == 'alice-typed-candidate-binding-v1'
                 and isinstance(self.binding['original_document_sha256'], str)
                 and re.fullmatch(r'[0-9a-f]{64}', self.binding['original_document_sha256'])
                 and self.binding['compiled_document_sha256'] == self.document.sha256
                 and self.binding['candidates_sha256'] == sha256(canonical([asdict(c) for c in self.candidates])).hexdigest()
                 and self.binding_sha256 == sha256(canonical(self.binding)).hexdigest(), 'typed candidate binding changed')


def compile_typed_candidates(document, candidates):
    document.validate()
    _require(type(candidates) is tuple and candidates and all(type(c) is TypedCandidate for c in candidates)
             and [c.candidate_id for c in candidates] == [c.candidate_id for c in document.candidates],
             'typed candidates must cover the complete ordered document')
    candidates = deepcopy(candidates)
    compiled = replace(document, candidates=tuple(replace(entry, text=candidate.semantic_text())
        for entry, candidate in zip(document.candidates, candidates)))
    binding = {'schema': 'alice-typed-candidate-binding-v1', 'original_document_sha256': document.sha256,
        'compiled_document_sha256': compiled.sha256,
        'candidates_sha256': sha256(canonical([asdict(c) for c in candidates])).hexdigest()}
    result = PreparedCandidates(compiled, candidates, binding, sha256(canonical(binding)).hexdigest())
    result.validate()
    return result


def respond_typed(consumer, prepared, produced, closed_session, *, expected_binding_sha256, renderer=None):
    _require(type(prepared) is PreparedCandidates, 'typed response needs the actual prepared candidates')
    prepared.validate()
    _require(prepared.binding_sha256 == expected_binding_sha256, 'typed candidate binding differs from external pin')
    respond = cumulative_successor.respond_with_governance if isinstance(consumer, cumulative_successor.SuccessorConsumer) \
        else governed_response.respond_with_governance
    governance = respond(consumer, prepared.document, produced, closed_session, renderer=None)
    selected = governance['selected_candidate_ids']
    result = {'schema': 'alice-personality-typed-handoff-v1', 'status': governance['status'],
        'typed_binding_sha256': expected_binding_sha256, 'governance': governance,
        'selected_candidate': None, 'speech_text': None, 'delivery': None,
        'text_position_unit': 'unicode_code_point', 'delivery_compatibility': 'UNREVIEWED',
        'disposition': 'UNRESOLVED', 'rendering': {'status': 'NOT_RENDERED'},
        'execution_performed': False, 'experience_recorded': False, 'acceptance_authority': False,
        'qualification': 'UNQUALIFIED', 'owner_fidelity_approved': False, 'spoken_fidelity_verified': False}
    if governance['status'] == 'SELECTED':
        _require(len(selected) == 1, 'typed handoff needs one actual governing selection')
        candidate = next(c for c in prepared.candidates if c.candidate_id == selected[0])
        result['selected_candidate'] = asdict(candidate)
        result['disposition'] = 'PROPOSAL_ONLY'
        if candidate.kind == 'utterance':
            intent = next(i for i in governance['preference_selection']['intents'] if i['candidate_id'] == selected[0])
            result['speech_text'] = candidate.payload['text']
            result['delivery'] = {'candidate_intent': deepcopy(candidate.payload['delivery']),
                                  'native_voice_controls': deepcopy(intent['voice_controls'])}
            result['disposition'] = 'UTTERANCE_INTENT'
            if renderer is not None:
                rendered = renderer(deepcopy(result))
                _require(type(rendered) is RenderedResponse, 'typed renderer must return actual explicit realization')
                _text(rendered.text)
                _text(rendered.renderer_id)
                _require(type(rendered.delivery) is dict, 'typed renderer must report its actual delivery')
                _json_data(rendered.delivery)
                _require(rendered.artifact_sha256 is None or isinstance(rendered.artifact_sha256, str)
                         and re.fullmatch(r'[0-9a-f]{64}', rendered.artifact_sha256), 'typed renderer artifact needs exact digest or absence')
                equivalent = rendered.text == result['speech_text'] and canonical(rendered.delivery) == canonical(result['delivery'])
                result['rendering'] = {'status': 'TEXT_REALIZED_DELIVERY_UNVERIFIED' if equivalent else 'UNRESOLVED_RENDERER',
                                       **deepcopy(vars(rendered))}
                if not equivalent:
                    result['status'] = 'UNRESOLVED_RENDERER'
    prepared.validate()
    _require(prepared.binding_sha256 == expected_binding_sha256, 'typed candidate binding changed during callback')
    _require(fingerprint(consumer.model.state_dict()) == governance['preference_selection']['model_state_sha256'],
             'typed governing model changed during callback')
    consumer._evidence()
    result['receipt_sha256'] = sha256(canonical(result)).hexdigest()
    return result
