# Security disclosure and data use

**This source code is intended for offline synthetic demonstrations and scoping discussions.** It is not certified for real banking, medical, law-enforcement, security incident or regulated customer data.

- Treat all outputs and locally generated reports as sensitive until you know their input origin. Do not commit customer files, `.env` files, local feedback, logs, or private reports to any GitHub repository.
- Local processing is not equivalent to secure deployment: the host operating system, backups, directory permissions, user accounts, retention and incident response are outside the prototype.
- The feedback registry has a conservative identifier-pattern screen, which does not guarantee anonymization or establish legal consent to process a comment.
- Do not use AA12–AA14 or AA16–AA18 with non-synthetic inputs. Other workflows require explicit authorization, binding contractual and technical controls before any real-data trial.
- Reports are advice for human review; the suite never approves payments, security incidents, legal notices, clinician decisions or regulatory filings.
- Before production: independent code/dependency/security review, secrets management, authentication and role control, access logging, encryption, backup/restore and deletion policies, privacy impact review, and deployment-specific threat modelling.

Report suspected vulnerabilities privately to an authorized AVAMALE maintainer; do not post actual customer data in issues.
