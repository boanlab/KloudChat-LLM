# Security policy

## Reporting a vulnerability

Report privately through
[GitHub Security Advisories](https://github.com/boanlab/KloudChat-LLM/security/advisories/new),
or by email to the maintainers. Do not open a public issue.

Include the affected component, the impact you were able to demonstrate, and
the steps to reproduce it. We aim to acknowledge a report within five working
days.

## Supported versions

Security fixes land on `main` and in the next tagged release. Older tags are
not patched.

## Threat model

This backend runs inside a private network. Three properties follow from that:

- **`/tools/*` is unauthenticated.** The gateway injects internal service
  credentials and ignores whatever the caller sent. Anything that can reach
  the gateway port can reach those tools.
- **`/tools/exec` runs arbitrary code.** The code interpreter is a sandbox
  with `SYS_ADMIN` and relaxed seccomp/apparmor so that user code can run. It
  is isolation, not a security boundary against a determined attacker.
- **LiteLLM's admin API and admin UI are reachable** at `/litellm/*`,
  authenticated by the master key only.

Exposing the gateway port to the internet, or to an untrusted segment of your
network, is a configuration mistake rather than a vulnerability in this
project. Reports of the form "the tools endpoint requires no authentication"
are expected behaviour and documented in [docs/tools.md](docs/tools.md).

Do report:

- A path through the gateway that reaches a service it should not, or that
  leaks an injected credential back to the caller.
- Sandbox escape from the code interpreter to the host or to another
  container.
- Credential exposure through generated config, logs, or the local ledger
  (`data/ledger/`), which stores issued keys in plaintext by design.
- Any way to make one user's requests bill to another user's budget.
