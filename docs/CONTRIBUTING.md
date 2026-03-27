# Contributing to Face Anti-Spoofing Project

Thank you for your interest in contributing to the Face Anti-Spoofing Detection project! This document provides guidelines and instructions for making meaningful contributions.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style Guidelines](#code-style-guidelines)
- [Commit Message Format](#commit-message-format)
- [Pull Request Process](#pull-request-process)
- [Testing](#testing)
- [Documentation](#documentation)
- [Reporting Issues](#reporting-issues)

## 🤝 Code of Conduct

This project adheres to the Contributor Covenant Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

**Be respectful, inclusive, and constructive** in all interactions with other contributors.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Git
- Basic understanding of Deep Learning
- Familiarity with PyTorch and PyTorch Lightning

### Initial Setup

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/face-anti-spoofing.git
   cd face-anti-spoofing
   ```

3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/dtruong46me/face-anti-spoofing.git
   ```

4. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 🔧 Development Setup

### Environment Setup

```bash
# Create conda environment
conda create -n fas-dev python=3.9
conda activate fas-dev

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e ".[dev]"
```

### Install Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

This ensures code quality checks run before each commit.

## 📝 Code Style Guidelines

### PEP 8 Compliance

This project follows [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines.

**Format code using Black:**
```bash
black src/ --line-length 100
black FAS_detector/ --line-length 100
```

**Lint code using Flake8:**
```bash
flake8 src/ --max-line-length 100 --count --statistics
flake8 FAS_detector/ --max-line-length 100 --count --statistics
```

### Type Hints

All functions should include type hints:

```python
from typing import Dict, List, Optional
import torch
from PIL import Image

def process_image(
    image: Image.Image,
    size: int = 224
) -> torch.Tensor:
    """
    Process an image and return tensor.
    
    Args:
        image: PIL Image object
        size: Target size for resizing
        
    Returns:
        Processed image tensor
    """
    # Implementation
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    epochs: int,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Train the FAS detection model.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        epochs: Number of training epochs
        device: Device to train on (cuda or cpu)
        
    Returns:
        Dictionary containing training metrics:
            - 'loss': Final training loss
            - 'accuracy': Training accuracy
            - 'acer': Average Classification Error Rate
            
    Raises:
        ValueError: If epochs is negative
        RuntimeError: If GPU memory is insufficient
        
    Example:
        >>> model = FeatherNetB()
        >>> metrics = train_model(model, train_loader, 40)
        >>> print(metrics['accuracy'])
    """
    pass
```

### Naming Conventions

- **Classes**: `CapitalizedWords` (PascalCase)
  ```python
  class FASPredictor:
      pass
  ```

- **Functions/Variables**: `lowercase_with_underscores` (snake_case)
  ```python
  def load_checkpoint(path: str) -> Dict:
      pass
  ```

- **Constants**: `UPPERCASE_WITH_UNDERSCORES`
  ```python
  DEFAULT_INPUT_SIZE = 224
  SUPPORTED_MODELS = ["feathernet", "mobilenetv3"]
  ```

- **Private Methods**: Prefix with underscore
  ```python
  def _preprocess_image(self, image: Image.Image):
      pass
  ```

## 📌 Commit Message Format

Use clear, descriptive commit messages. Follow this format:

```
<type>(<scope>): <subject>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```

### Type

Must be one of:
- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, missing semicolons, etc.)
- **refactor**: Code refactoring without feature/fix changes
- **perf**: Performance improvements
- **test**: Adding or updating tests
- **chore**: Changes to build system or dependencies

### Scope

Optional, specifies which component is affected:
- `predictor`
- `training`
- `models`
- `metrics`
- `config`
- `docs`

### Subject

- Use imperative mood ("add feature" not "added feature")
- Don't capitalize first letter
- No period at the end
- Max 50 characters

### Body

- Explain what and why, not how
- Wrap at 72 characters
- Reference issues: `Fixes #123`, `Related to #456`

### Examples

```
feat(predictor): add batch prediction support

Added ability to run predictions on multiple images efficiently.
This includes memory optimization for large batches.

Fixes #42
```

```
fix(training): handle empty batches in dataloader

Previous implementation crashed when encountering empty batches.
Now properly skips empty batches and logs warning.
```

## 🔄 Pull Request Process

### Before Submitting

1. **Update your branch** with latest changes from upstream:
   ```bash
   git fetch upstream
   git rebase upstream/master
   ```

2. **Run tests** and code quality checks:
   ```bash
   pytest tests/ -v
   flake8 src/
   black src/ --check
   mypy src/ --ignore-missing-imports
   ```

3. **Commit your changes** with proper messages

4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

### PR Description Template

```markdown
## Description
Brief description of the changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Related Issues
Fixes #123

## Testing
- [ ] Added unit tests
- [ ] Tested with pre-trained models
- [ ] Tested on GPU/CPU

## Checklist
- [ ] Code follows style guidelines
- [ ] Added/updated documentation
- [ ] Commit messages are clear
- [ ] No new warnings generated
- [ ] Dependencies updated in requirements.txt
```

### PR Review Expectations

- Respond to review comments promptly
- Make requested changes in new commits (don't force push)
- Request re-review once changes are addressed
- Be open to constructive feedback

## 🧪 Testing

### Writing Tests

Create tests in `tests/` directory matching the module structure:

```python
# tests/test_predictor.py
import pytest
from PIL import Image
from src.predictor import FASPredictor


class TestFASPredictor:
    """Test FAS predictor functionality."""
    
    @pytest.fixture
    def predictor(self):
        """Initialize predictor for testing."""
        return FASPredictor(
            model_checkpoint="path/to/test/model.ckpt",
            modelname="feathernet"
        )
    
    def test_predict_real_face(self, predictor):
        """Test prediction on real face image."""
        image = Image.open("tests/data/real_face.jpg")
        result = predictor.predict(image)
        
        assert result['label'] == 'real'
        assert 0 <= result['probability'] <= 1
        assert result['tensor'] == [1, 0]
    
    def test_predict_fake_face(self, predictor):
        """Test prediction on fake/spoofed face."""
        image = Image.open("tests/data/fake_face.jpg")
        result = predictor.predict(image)
        
        assert result['label'] == 'fake'
        assert 0 <= result['probability'] <= 1
        assert result['tensor'] == [0, 1]
    
    @pytest.mark.parametrize("image_path", [
        "tests/data/real_1.jpg",
        "tests/data/real_2.jpg",
        "tests/data/real_3.jpg"
    ])
    def test_batch_predict(self, predictor, image_path):
        """Test batch prediction."""
        images = [image_path] * 3
        results = predictor.batch_predict(images)
        
        assert len(results) == 3
        for result in results:
            assert 'label' in result
            assert 'probability' in result
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_predictor.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run with markers
pytest tests/ -m "not slow" -v
```

## 📚 Documentation

### Updating README

When adding features, update the relevant section in README.md with examples and explanations.

### Adding Module Documentation

Create/update docstrings and add to documentation:

```python
"""
predictor.py - FAS Prediction Module

This module provides the FASPredictor class for running inference
on face images to detect whether they are real or spoofed.

Example:
    >>> from src.predictor import FASPredictor
    >>> predictor = FASPredictor(
    ...     model_checkpoint="model.ckpt",
    ...     modelname="feathernet"
    ... )
    >>> result = predictor.predict_from_path("face.jpg")
"""
```

## 🐛 Reporting Issues

### Issue Template

```markdown
## Description
Clear description of the issue.

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen.

## Actual Behavior
What actually happens.

## Environment
- OS: [e.g., Windows 10, Ubuntu 20.04]
- Python: 3.9
- PyTorch: 2.0
- GPU: [e.g., NVIDIA A100, CPU only]

## Logs/Screenshots
Attach any error logs or screenshots.
```

### Finding Issues to Work On

Look for issues labeled:
- `good first issue` - Great for newcomers
- `help wanted` - Need assistance
- `bug` - Known issues to fix
- `enhancement` - Feature requests

## 📞 Getting Help

- **Discussions**: Use GitHub Discussions for questions
- **Issues**: Report bugs with detailed information
- **Email**: Contact project maintainers for sensitive issues

## 🎯 Development Priorities

Currently seeking contributions in these areas:

1. **Model Optimization**: Quantization, pruning, distillation
2. **Performance**: Improve inference speed and memory usage
3. **Documentation**: Expand tutorials and examples
4. **Testing**: Increase test coverage
5. **Deployment**: ONNX export, TorchScript support
6. **Dataset Tools**: Data augmentation, preprocessing utilities

## ✅ Review Checklist

Before marking PR as ready for review, ensure:

- [ ] Code follows project style guidelines
- [ ] All tests pass locally
- [ ] Code changes are well-documented
- [ ] Commit messages are clear and descriptive
- [ ] No unnecessary files added (check .gitignore)
- [ ] Dependencies are documented
- [ ] Breaking changes are clearly noted
- [ ] Examples/demos updated if needed

## 📜 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to Face Anti-Spoofing Detection!** 🎉

For questions or issues with the contribution process, feel free to open a discussion or contact the project maintainers.
