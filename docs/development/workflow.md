# Development Workflow

This project follows GitHub Flow, a lightweight, branch-based workflow.

## GitHub Flow

1. **Main Branch**
   - The `main` branch is always deployable
   - All feature development and fixes branch from `main`
   - Merges to `main` trigger deployments

2. **Feature Branches**
   - Create a branch for each feature/fix
   - Use descriptive names with prefixes:
     - `feature/` for new features
     - `fix/` for bug fixes
     - `docs/` for documentation
     - `refactor/` for code refactoring

3. **Development Process**
   ```bash
   # Create a new branch
   git checkout -b feature/new-feature

   # Make changes and commit
   git add .
   git commit -m "feat: add new feature"

   # Push changes
   git push origin feature/new-feature
   ```

4. **Pull Requests**
   - Create a PR when your feature is ready
   - Fill in the PR template
   - Request reviews from team members
   - Address review comments

5. **CI/CD Pipeline**
   - Automated checks run on every PR:
     - Code linting
     - Tests
     - Documentation build
   - Successful checks required for merge

6. **Deployment**
   - Merges to `main` trigger:
     - Docker image build and push
     - Documentation deployment
     - Package publication

## Best Practices

1. **Commits**
   - Use conventional commit messages
   - Keep commits focused and atomic
   - Write clear commit descriptions

2. **Code Quality**
   - Run linters locally before pushing
   - Add tests for new features
   - Update documentation

3. **Reviews**
   - Review PRs promptly
   - Provide constructive feedback
   - Test changes locally if needed

4. **Documentation**
   - Update docs with code changes
   - Keep README current
   - Add inline code comments 