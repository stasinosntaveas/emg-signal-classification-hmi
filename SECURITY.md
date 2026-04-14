# Security Policy

## Supported Versions

This project is a research/academic machine learning system for EMG signal classification. Security updates are generally applied only to the latest version of the repository.

| Version | Supported |
|--------|-----------|
| main   | ✅        |
| older versions | ❌ |

---

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it responsibly.

You can report issues via:
- GitHub Issues (preferred for non-sensitive reports)
- Direct contact (if applicable for sensitive issues)

Please include:
- Description of the issue
- Steps to reproduce
- Potential impact
- Suggested fix (if available)

---

## Scope

This security policy applies to:
- Streamlit application (`src/app.py`)
- Model loading and inference pipeline
- Data preprocessing and feature extraction
- Serialized model files (`.pkl`)

---

## Out of Scope

The following are not considered security vulnerabilities:
- Issues in third-party libraries (e.g., scikit-learn, Streamlit, SciPy)
- Incorrect model predictions or low accuracy
- Dataset-related biases or labeling errors
- Performance limitations or slow execution

---

## Security Considerations

This project uses:
- File uploads (`.mat` files via Streamlit)
- Deserialization of pre-trained models (`joblib`)
- Local execution of Python code for inference

To reduce risk:
- Only upload trusted `.mat` files
- Do not load untrusted `.pkl` model files
- Avoid exposing the Streamlit app publicly without authentication
- Keep dependencies updated

---

## Data Safety

- The system does not transmit data externally
- All processing is performed locally
- Uploaded files are not stored permanently unless modified in code

---

## Responsible Use

This project is intended for:
- Academic research
- Signal processing experiments
- Machine learning model development

It is not designed for production medical or safety-critical use.

---

## Contact

For security-related concerns, open an issue or contact the maintainer through GitHub.
