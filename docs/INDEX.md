# Documentation Index

Welcome to the Face Anti-Spoofing documentation! This page provides navigation to all available documentation.

---

## 📖 Quick Links

### Getting Started
- **[Setup Guide](SETUP.md)** - Installation, environment setup, model placement, dataset organization, troubleshooting
- **[API Documentation](API_DOCUMENTATION.md)** - Complete API reference for all modules and classes

### Contributing & Development
- **[Contributing Guidelines](CONTRIBUTING.md)** - How to contribute, coding standards, PR process, testing guidelines

### Refactoring & History
- **[Refactoring Summary](REFACTORING_SUMMARY.md)** - Detailed refactoring changes and improvements
- **[Refactoring Complete](REFACTORING_COMPLETE.txt)** - Quick summary of all refactoring work

---

## 📚 Documentation Overview

### [SETUP.md](SETUP.md)
Complete guide for setting up the project. Includes:
- System requirements (hardware and software)
- Directory structure setup
- 4 Installation methods (Conda, venv, pip, Docker)
- Model setup and downloading pre-trained weights
- Dataset organization (LCC-FASD structure)
- Test images setup
- Verification and quick start
- Environment variables for GPU optimization
- 9 Troubleshooting scenarios with solutions
- Configuration files reference

**Best for**: First-time setup, installation help, troubleshooting issues

### [API_DOCUMENTATION.md](API_DOCUMENTATION.md)
Complete API reference for the Face Anti-Spoofing framework. Includes:
- **Core Module**: `src.predictor` - Main prediction interface
- **Configuration**: `src.config` - Configuration management
- **Models**: `src.models` - All supported model architectures
- **Data**: `src.data` - Dataset loading and preprocessing
- **Metrics**: `src.metrics` - Evaluation metrics (APCER, NPCER, ACER, Accuracy, Recall)
- **Pipelines**: `src.pipelines` - Training pipelines
- **Detection**: `FAS_detector` - Real-time video detection
- Usage examples (single image, batch processing, video detection, training)
- Return value formats
- Error handling guide
- Performance optimization tips

**Best for**: API usage, understanding module structure, code examples

### [CONTRIBUTING.md](CONTRIBUTING.md)
Guidelines for contributing to the project. Includes:
- Code of Conduct
- Getting Started (prerequisites, initial setup)
- Development Setup (environment, pre-commit hooks)
- Code Style Guidelines:
  - PEP 8 compliance
  - Type hints requirements
  - Google-style docstrings
  - Naming conventions
- Commit Message Format with examples
- Pull Request Process
- Testing Guidelines
- Documentation updates
- Issue reporting
- Development priorities

**Best for**: Contributing code, understanding project standards, submitting PRs

---

## 🚀 Common Workflows

### Want to install the project?
→ Start with [SETUP.md](SETUP.md)

### Need to use the API?
→ Check [API_DOCUMENTATION.md](API_DOCUMENTATION.md)

### Want to contribute?
→ Read [CONTRIBUTING.md](CONTRIBUTING.md)

### Have a question about specific modules?
→ See [API_DOCUMENTATION.md](API_DOCUMENTATION.md#table-of-contents)

### Running into issues?
→ Check [SETUP.md - Troubleshooting](SETUP.md#-troubleshooting)

---

## 📋 Content Summary

| Document | Purpose | Size | Topics |
|----------|---------|------|--------|
| SETUP.md | Installation & Configuration | 800+ lines | Requirements, Installation, Setup, Verification, Troubleshooting |
| API_DOCUMENTATION.md | API Reference | 500+ lines | Modules, Classes, Functions, Examples, Error Handling |
| CONTRIBUTING.md | Project Contribution | 450+ lines | Code standards, Testing, PR process, Issue reporting |

---

## 🔍 Key Topics by Document

### SETUP.md
- ✅ Installation methods (4 options)
- ✅ GPU/CUDA setup
- ✅ Model downloading
- ✅ Dataset organization
- ✅ Verification steps
- ✅ Troubleshooting

### API_DOCUMENTATION.md
- ✅ `FASPredictor` usage
- ✅ Model architecture details
- ✅ Data loading pipeline
- ✅ Metrics definitions
- ✅ Code examples
- ✅ Performance tips

### CONTRIBUTING.md
- ✅ PEP 8 standards
- ✅ Type hints requirements
- ✅ Docstring format (Google-style)
- ✅ Commit message format
- ✅ PR template and process
- ✅ Testing guidelines

---

## 💡 Document Navigation

**Each document includes:**
- 📋 Table of Contents at the top
- 🔍 Detailed sections with examples
- 🔗 Cross-references to other docs
- 📝 Code examples where applicable

---

## 📞 Support

If you can't find what you're looking for:
1. Check the Table of Contents in each document
2. Use Ctrl+F (or Cmd+F) to search within documents
3. Check the troubleshooting section in [SETUP.md](SETUP.md)
4. Open an issue on GitHub with details

---

## 🗂️ Documentation Structure

```
docs/
├── INDEX.md                    ← You are here
├── SETUP.md                    ← Installation & Setup
├── API_DOCUMENTATION.md        ← API Reference
└── CONTRIBUTING.md             ← Contributing Guidelines
```

**Root level:**
- `README.md` - Project overview and quick start

---

**Last Updated**: 2024
**Version**: 1.0
