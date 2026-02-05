# PIMALUOS Maintenance Plan

## Project Governance

### Maintainers
- **Core Team:** Responsible for strategic direction, major releases, and code review
- **Contributors:** Community members who submit pull requests and improvements
- **Users:** Researchers and practitioners providing feedback and use cases

### Decision Making
- **Minor changes:** Single maintainer approval
- **Major changes:** Consensus among core team
- **Breaking changes:** Community discussion via GitHub Discussions

## Release Cycle

### Versioning
We follow [Semantic Versioning](https://semver.org/):
- **MAJOR:** Breaking API changes
- **MINOR:** New features, backward compatible
- **PATCH:** Bug fixes, backward compatible

### Release Schedule
- **Patch releases:** As needed for critical bugs
- **Minor releases:** Quarterly (Jan, Apr, Jul, Oct)
- **Major releases:** Annually or when significant breaking changes accumulate

## Dependency Management

### Strategy
1. **Pin major versions** in requirements.txt (e.g., `torch>=2.0.0,<3.0.0`)
2. **Test against latest** minor versions in CI
3. **Update dependencies** quarterly
4. **Security patches:** Apply immediately

### Critical Dependencies

#### Deep Learning Stack
- **PyTorch:** Follow PyTorch LTS releases
- **PyTorch Geometric:** Update within 3 months of PyTorch updates
- **Plan:** Maintain compatibility with PyTorch N and N-1 versions

#### LLM APIs
- **OpenAI:** Monitor API changes, maintain backward compatibility
- **Anthropic:** Same as OpenAI
- **Mitigation:** Abstract LLM interface allows swapping providers
- **Local fallback:** Ollama support ensures independence from cloud APIs

#### Geospatial Libraries
- **GeoPandas:** Stable, update conservatively
- **Shapely:** Follow GeoPandas compatibility
- **Plan:** Test against GDAL/PROJ updates

## Long-term Sustainability

### API Stability
- **Public API:** Maintain backward compatibility for 2 major versions
- **Deprecation policy:** 
  1. Mark as deprecated in version N
  2. Warn in version N+1
  3. Remove in version N+2
- **Documentation:** Maintain migration guides

### LLM API Evolution

**Challenge:** Cloud LLM APIs may change pricing, deprecate models, or shut down.

**Mitigation Strategies:**

1. **Provider Abstraction**
   ```python
   # Users can swap providers without code changes
   llm = get_llm('openai')  # or 'anthropic', 'ollama', 'mock'
   ```

2. **Local-First Option**
   - Ollama support for fully local operation
   - Mock LLM for testing without APIs
   - Pre-extracted constraints cached for common zones

3. **Constraint Caching**
   - Cache extracted zoning constraints
   - Reduce API calls by 90%+ for repeated queries
   - Share constraint databases via Zenodo

4. **Community Constraint Library**
   - Build open database of extracted zoning rules
   - Reduce dependency on LLM APIs over time
   - Enable offline operation for common cities

### Infrastructure

**Code Hosting:** GitHub (with GitLab mirror)  
**Documentation:** ReadTheDocs (self-hosted fallback available)  
**Package Distribution:** PyPI (conda-forge planned)  
**Data Archives:** Zenodo (permanent DOI)  
**CI/CD:** GitHub Actions (self-hosted runners available)

### Funding & Support

**Current:** Academic research project  
**Future Options:**
- Grant funding for continued development
- Institutional partnerships
- Community sponsorship (GitHub Sponsors)
- Consulting services for custom deployments

### Community Building

**Communication Channels:**
- GitHub Discussions for Q&A
- Quarterly community calls
- Annual user survey
- Conference presentations (AAG, CUPUM, CEUS)

**Documentation:**
- Maintain comprehensive tutorials
- Video walkthroughs
- Example notebooks
- API reference

**Training:**
- Workshop materials
- Online course (planned)
- Webinar series

## Succession Planning

### Knowledge Transfer
- **Documentation:** All design decisions documented
- **Code comments:** Explain "why" not just "what"
- **Architecture docs:** High-level system design
- **Onboarding guide:** For new maintainers

### Maintainer Rotation
- Encourage contributor → maintainer pipeline
- Pair programming for knowledge sharing
- Regular code review participation
- Gradual responsibility increase

## Monitoring & Metrics

### Health Indicators
- **Test coverage:** Maintain >80%
- **Issue response time:** <7 days
- **PR review time:** <14 days
- **Documentation coverage:** All public APIs
- **Download stats:** Track via PyPI
- **Citation count:** Monitor via Google Scholar

### Quarterly Review
- Dependency updates
- Security audit
- Performance benchmarks
- User feedback analysis
- Roadmap adjustment

## Roadmap (2025-2027)

### 2025 Q1-Q2
- ✅ Initial release (v0.1.0)
- 🔄 CEUS journal publication
- 📚 Documentation expansion
- 🧪 Test coverage to 90%

### 2025 Q3-Q4
- 🌍 Add 3 more cities (Chicago, LA, Boston)
- 🎨 Dashboard v2 with improved UX
- 🔌 QGIS plugin (alpha)
- 📦 Conda-forge package

### 2026
- 🌐 International cities (London, Tokyo, Singapore)
- 🤖 Fine-tuned local LLM for zoning
- ⚡ GPU optimization for 100K+ parcels
- 🎓 Online course launch

### 2027
- 🏛️ Integration with official planning systems
- 🌊 Real-time digital twin capabilities
- 🤝 Multi-city regional optimization
- 📊 Impact assessment framework

## Contact

**Maintainer Team:** pimaluos@example.com  
**Security Issues:** security@pimaluos.org  
**General Questions:** GitHub Discussions

---

*Last Updated: January 2026*  
*Next Review: April 2026*
