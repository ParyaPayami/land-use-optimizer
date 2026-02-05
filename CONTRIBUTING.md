# Contributing to PIMALUOS

Thank you for your interest in contributing to PIMALUOS! This document provides guidelines for contributing to the project.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and professional in all interactions.

## How to Contribute

### Reporting Issues

- Use GitHub Issues to report bugs or request features
- Search existing issues before creating a new one
- Provide detailed information: OS, Python version, error messages, minimal reproducible example

### Pull Requests

1. **Fork the repository** and create a feature branch
2. **Make your changes** following our coding standards
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Run tests** and ensure they pass
6. **Submit a pull request** with a clear description

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/PIMALUOS.git
cd PIMALUOS

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Coding Standards

### Style Guide

- **Python:** Follow PEP 8
- **Line length:** 100 characters (configured in black)
- **Imports:** Organized with isort
- **Type hints:** Use for function signatures
- **Docstrings:** Google style

### Code Quality Tools

```bash
# Format code
black pimaluos/
isort pimaluos/

# Lint
flake8 pimaluos/
pylint pimaluos/

# Type check
mypy pimaluos/
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=pimaluos --cov-report=html

# Run specific test file
pytest tests/test_core.py -v
```

## Documentation

- Update docstrings for new functions/classes
- Add examples to `examples/` directory
- Update `docs/` for major features
- Keep README.md current

## Areas for Contribution

### High Priority
- 🌍 **Multi-city support:** Add data loaders for new cities
- 🧪 **Testing:** Increase test coverage
- 📚 **Documentation:** Tutorials, examples, API docs
- 🐛 **Bug fixes:** Check GitHub Issues

### Medium Priority
- ⚡ **Performance:** Optimize graph construction, GPU utilization
- 🎨 **Dashboard:** UI/UX improvements
- 🔌 **Integrations:** QGIS plugin, ArcGIS compatibility
- 🌐 **Internationalization:** Support for non-English zoning codes

### Advanced
- 🧠 **New models:** Alternative GNN architectures
- 🤖 **Agent types:** New stakeholder profiles
- ⚙️ **Physics engines:** Air quality, noise, energy
- 🎯 **Optimization:** New algorithms, constraint handling

## Commit Messages

Use conventional commits format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(core): add Chicago data loader
fix(physics): correct BPR function parameters
docs(readme): update installation instructions
```

## Review Process

1. Automated checks must pass (tests, linting)
2. At least one maintainer review required
3. Address review comments
4. Maintainer merges when approved

## Questions?

- 💬 **Discussions:** Use GitHub Discussions for questions
- 📧 **Email:** pimaluos@example.com
- 📖 **Documentation:** https://pimaluos.readthedocs.io

Thank you for contributing to open urban data science! 🏙️
