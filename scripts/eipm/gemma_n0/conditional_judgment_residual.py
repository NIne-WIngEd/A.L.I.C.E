"""Experimental N2 conditional access; never an implicitly admitted native export.

The legacy N2 modules and state keys remain intact. The explicit extension feeds
all existing heads and uses only native authorized views. No upstream mutation.
"""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import torch
from torch import nn

from src.alice_personality.identity.model import IdentityModel, IdentityJudgmentModel, IdentityRepresentationModel
from src.alice_personality.identity.codec import fingerprint

ARCHITECTURE = 'alice-personality-conditional-judgment-residual-v1'


class ConditionalResidual(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.gate = nn.Parameter(torch.zeros(()))
        self.register_buffer('enabled',torch.tensor(True))

    def forward(self, candidates, candidate_mask, memory, memory_mask):
        # An explicit zero sentinel keeps an empty authorized row finite; it is
        # removed from the result, including attention projection biases.
        if memory.shape[1] == 0:
            memory = candidates.new_zeros(candidates.shape[0],1,candidates.shape[-1])
            memory_mask = torch.zeros(candidates.shape[0],1,dtype=torch.bool,device=candidates.device)
        available = memory_mask.any(-1)
        safe_mask = memory_mask.clone()
        safe_mask[~available,0] = True
        safe_memory = memory.masked_fill(~memory_mask[...,None],0)
        attended,_ = self.attention(self.query_norm(candidates),safe_memory,safe_memory,
                                   key_padding_mask=~safe_mask,need_weights=False)
        return attended * (self.gate.tanh()*self.enabled) * (candidate_mask & available[:,None])[...,None]


class ConditionalJudgment(IdentityJudgmentModel):
    """Versioned experiment mirroring the original N2 forward plus one residual.

No native config/loader is changed. A strict legacy load rejects the added keys.
The zero-gate regression checks full packets and original-parameter derivatives.
"""
    def __init__(self, config):
        super().__init__(config)
        self.conditioning = ConditionalResidual(config.learned_width,config.attention_heads)

    def conditioning_context(self,n1):
        _,views = self.conditional_memory(n1)
        return torch.cat(tuple(views.values()),1),torch.cat(tuple(n1.view_masks.values()),1)

    def forward(self,frame,n1,query):
        frame.validate(self.config)
        _,candidates,_ = self.readout(frame.candidates,query)
        mask = frame.candidates.entry_mask & frame.candidate_authority_mask & frame.candidate_context_mask
        tokens = n1.source_tokens.flatten(1,2)
        source_available = torch.stack(tuple(IdentityRepresentationModel.source_masks(frame).values())).any(0)
        token_mask = (frame.sources.token_mask & source_available[...,None]).flatten(1,2)
        contextual_concepts,contextual_views = self.conditional_memory(n1)
        memory = torch.cat((n1.query_tokens,tokens,*contextual_views.values()),1)
        memory_mask = torch.cat((n1.query_mask,token_mask,*n1.view_masks.values()),1)
        for layer in self.layers:
            candidates = layer(candidates,mask,memory,memory_mask)
        # Separate normalized attention domain, without normalizing away the
        # source/concept magnitude or replacing full query/source-token access.
        condition = torch.cat(tuple(contextual_views.values()),1)
        condition_mask = torch.cat(tuple(n1.view_masks.values()),1)
        candidates = candidates + self.conditioning(candidates,mask,condition,condition_mask)
        heads = {};label_states = {}
        for family,bank in frame.labels.items():
            _,labels,_ = self.readout(bank,query)
            label_states[family] = labels
            heads[family] = self.heads['family_'+family](candidates,labels)
        evidence_logits = torch.einsum('bch,bsh->bcs',self.evidence_query(candidates),self.evidence_key(n1.source_states))
        concept_logits = torch.einsum('bch,bkh->bck',self.concept_query(candidates),self.concept_key(contextual_concepts))
        return {'candidates':candidates,'candidate_mask':mask,'heads':heads,'scalar':self.scalar(candidates),
                'evidence_logits':evidence_logits,'concept_logits':concept_logits,'source_mask':source_available,
                'voice_confidence':self.voice_confidence(candidates,label_states['voice'])}


def migrate(model,*,seed):
    if type(model) is not IdentityModel or type(model.n2) is not IdentityJudgmentModel:
        raise ValueError('exact original native model required for explicit migration')
    if model.phase != 'inference' or model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('frozen original inference state required')
    if type(seed) is not int or seed < 0:
        raise ValueError('explicit nonnegative initialization seed required')
    before = fingerprint(model.state_dict())
    parameter = next(model.parameters())
    devices = [] if parameter.device.type != 'cuda' else [parameter.device.index]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        result = deepcopy(model)
        n2 = ConditionalJudgment(model.config).to(device=parameter.device,dtype=parameter.dtype)
        missing = n2.load_state_dict(model.n2.state_dict(),strict=False)
        expected = {key for key in n2.state_dict() if key.startswith('conditioning.')}
        if set(missing.missing_keys) != expected or missing.unexpected_keys:
            raise ValueError('original learned tensors were not retained exactly')
        result.n2 = n2
        result.set_phase('inference')
    if fingerprint(model.state_dict()) != before or any(not torch.equal(value,result.state_dict()[key]) for key,value in model.state_dict().items()):
        raise ValueError('migration changed compatible original state')
    receipt = {'architecture':ARCHITECTURE,'extension_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
               'initialization_seed':seed,'parent_model_state_sha256':before,
               'experimental_model_state_sha256':fingerprint(result.state_dict()),
               'original_tensors_retained_exactly':True,'initial_gate':0.,'native_checkpoint':False,
               'qualification':'UNQUALIFIED','acceptance_authority':False}
    return result,receipt
