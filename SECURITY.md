# Security policy

## Reporting a vulnerability

Report privately through
[GitHub Security Advisories](https://github.com/boanlab/KloudChat-LLM/security/advisories/new),
or by email to the maintainers. Do not open a public issue.

Include the affected component, the impact you were able to demonstrate, and
the steps to reproduce it. Reports are acknowledged within five working days.

## Supported versions

Security fixes land on `main` and in the next tagged release. Older tags are
not patched.

## Threat model

This backend runs inside a private network. Three properties follow from that:

- **`/tools/*` requires one shared key.** The gateway answers `401` unless the
  request carries `X-Tools-Key: <TOOLS_GATEWAY_KEY>`, then injects internal
  service credentials and ignores whatever else the caller sent. The key is a
  single shared secret held by KloudChat's API, not a per-user credential:
  whoever holds it can reach every tool. See
  [docs/tools.md](docs/tools.md#authentication).
- **`/tools/exec` runs arbitrary code.** The code interpreter is a sandbox
  with `SYS_ADMIN` and relaxed seccomp/apparmor so that user code can run. It
  is isolation, not a security boundary against a determined attacker.
- **LiteLLM's admin API and admin UI are reachable** at `/litellm/*`,
  authenticated by the master key only.

Exposing the gateway port to the internet, or to an untrusted segment of your
network, is a configuration mistake rather than a vulnerability in this
project; the tools key is a second line of defence, not a reason to publish
the port.

Do report:

- A path through the gateway that reaches a service it should not, reaches
  `/tools/*` without the tools key, or leaks an injected credential (or the
  tools key) back to the caller or to an upstream.
- Sandbox escape from the code interpreter to the host or to another
  container.
- Credential exposure through generated config, logs, or the local ledger
  (`data/ledger/`), which stores issued keys in plaintext by design.
- Any way to make one user's requests bill to another user's budget.
