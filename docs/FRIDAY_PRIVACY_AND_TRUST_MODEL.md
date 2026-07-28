# Friday Privacy and Trust Model

## 1. Goal

The product claim is stronger than "runs locally." Friday must make it structurally difficult for the developer, an update service, a plugin, or another local user to read or exfiltrate the host's personal intelligence.

## 2. Threat actors

- Friday developers or compromised vendor infrastructure;
- malicious update or dependency;
- malicious skill/plugin;
- another local operating-system account;
- malware with user-level or administrator access;
- remote model or connector provider;
- stolen device;
- accidental telemetry or support bundle;
- prompt-injection content attempting to change memory or cause exfiltration.

## 3. Trust principles

### Host-held keys

The vendor does not possess a universal decryption key. Installation generates local key material. Recovery is host-controlled.

### Content-free vendor plane

Licensing, updates, and package delivery operate without personal content. Telemetry must not include prompts, responses, file names, embeddings, memory, adapter weights, or extracted text by default.

### Local encryption domains

Separate keys or derivation contexts protect:

- original sources;
- extracted text and media derivatives;
- memory and beliefs;
- credentials;
- adapters and training data;
- audit logs and Identity Capsules.

### Visible egress

The host can see which component sent data, destination, purpose, and data class. Offline mode blocks all non-local destinations.

### Lifecycle and backup custody

Hot, warm, cold, quarantine, and backup tiers remain inside host-controlled encryption domains. Moving data to an external drive, NAS, or optional cloud archive does not grant the vendor access. Off-device archives use end-to-end encryption with host-held keys and expose their destination, retention, restore, and purge behavior. Cross-host deduplication is prohibited because shared content hashes can leak equality information between users.

### Least vendor knowledge, not least local intelligence

Friday may learn extensively about its host. The privacy objective is to keep that intelligence under host custody, not to minimize useful personalization.

## 4. Developer non-access claim

The public product claim should be:

> Friday's default architecture does not send the developer your personal files, memories, embeddings, conversations, or personalized model state, and the developer does not hold the keys needed to decrypt them.

Do not claim absolute impossibility against a compromised operating system, administrator malware, or a malicious binary update. Instead provide verifiable controls, signed releases, source review, and egress transparency.

## 5. Update security

- signed application and update packages;
- pinned update public keys;
- staged channels;
- rollback support;
- dependency manifests and software bill of materials;
- update diff or change summary for sensitive components;
- no update may export local data as a migration prerequisite;
- high-impact migrations create a local snapshot first.

## 6. Skill/plugin isolation

Every skill declares:

- file scopes;
- memory scopes;
- network destinations;
- executable permissions;
- credential access;
- background behavior;
- retention behavior.

The runtime enforces these declarations. A plugin cannot gain blanket access simply because Friday itself has it.

## 7. Deletion and unlearning

Deletion must propagate through:

- source storage;
- indexes and caches;
- memories and beliefs;
- derived training examples;
- active adapters where technically supported;
- future retraining queues;
- exported capsules created after deletion.

Where exact parameter unlearning cannot be guaranteed, Friday must say so and offer adapter deletion and retraining from retained lineage. Backup and cold-archive deletion follows a declared purge schedule; new restores and exports must not reintroduce payloads whose deletion lineage is active.

## 8. Optional services

Optional backup, sync, remote inference, and federated improvement are separate capabilities. Each requires an explicit host-selected mode and documented information flow. The local product remains functional without them.
