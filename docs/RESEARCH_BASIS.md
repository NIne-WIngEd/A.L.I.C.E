# Research Basis for A.L.I.C.E. Governance 1.0

The architecture draws from primary research demonstrating that agents can accumulate executable skills, learn from environmental feedback, self-modify code, conduct increasingly autonomous scientific work, and improve through empirical evaluation.

Key references:

- Wang et al., *Voyager: An Open-Ended Embodied Agent with Large Language Models*, arXiv:2305.16291.
- Zhang et al., *Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents*, arXiv:2505.22954.
- Weng et al., *A Self-Improving Coding Agent*, arXiv:2504.15228.
- Yamada et al., *The AI Scientist-v2*, arXiv:2504.08066.
- Xie et al., *OSWorld*, arXiv:2404.07972; and Yuan et al., *OSWorld 2.0*, arXiv:2606.29537.
- Debenedetti et al., *AgentDojo*, arXiv:2406.13352.
- Shumailov et al., *AI models collapse when trained on recursively generated data*, Nature 631, 2024.
- Bai et al., *Constitutional AI*, arXiv:2212.08073.

The design conclusion is capability-first but evidence-driven: broad research freedom, automated learning, and self-evolution require strong evaluators, provenance, containment, and rollback because those mechanisms make improvement real rather than merely claimed.

## Friday product and local-personalization basis

- Magister et al., *On the Way to LLM Personalization: Learning to Remember User Conversations*, Apple ML Research, 2025. Demonstrates parameter-efficient conversational personalization and reports competitive results using LoRA.
- Qin et al., *Enabling On-Device Large Language Model Personalization with Self-Supervised Data Selection and Synthesis*, arXiv:2311.12275. Proposes local representative-data selection and on-device personalization under limited storage and annotations.
- ONNX Runtime, *On-Device Training* documentation. Establishes a supported path for training smaller components on user devices.
- llama.cpp official releases and backends. Provides portable quantized inference across Windows x64, ARM64, CUDA, Vulkan, OpenVINO, SYCL, and Qualcomm Adreno/OpenCL targets.
- Microsoft, Windows DPAPI and Windows application packaging/code-signing documentation. Supports OS-bound key protection and signed Windows distribution.
- Tauri 2 documentation. Supports Windows EXE/MSI packaging and signed updater workflows for a web-technology desktop shell.
- Chung and Badhe, *Local Is Not a Sufficient Privacy Boundary: Governing OS-Integrated On-Device AI*, arXiv:2606.10173. Motivates visible information flow, bounded vendor access, and lifecycle auditing beyond the simple claim that inference is local.
- Apple, *Private Cloud Compute Security Guide*. Demonstrates the product value of externally inspectable privacy architecture rather than a policy-only promise.

## Competitive references

- Jan official product site: local/open personal intelligence and planned memory.
- LM Studio official product site: native local models and agentic workflows.
- Personal.ai official product materials: per-user memory stacks, personal language models, continuous training, and Windows/Mac applications.
- Wu et al., *OS-Copilot: Towards Generalist Computer Agents with Self-Improvement*, arXiv:2402.07456. Uses the name FRIDAY for an academic self-improving computer agent, creating a public-name collision that requires clearance.

These references mean Friday cannot rely on "local" or "personalized" as a standalone moat. The planned differentiation is the complete combination of local multimodal ingestion, selective lifelong learning, inspectable beliefs, host-owned learned parameters, portable identity, autonomous skills, and proof-oriented developer non-access.
