#!/bin/bash
set -e

echo "🚀 Publishing gac to PyPI..."

# Install build tools if needed
echo "📦 Installing build tools..."
uv pip install build twine

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info

# Build package
echo "🔨 Building package..."
python -m build

# Upload to PyPI
echo "📤 Uploading to PyPI..."
echo ""
echo "Choose upload destination:"
echo "1) TestPyPI (for testing)"
echo "2) PyPI (production)"
read -p "Enter choice (1 or 2): " choice

if [ "$choice" = "1" ]; then
    echo "Uploading to TestPyPI..."
    twine upload --repository testpypi dist/*
    echo "✅ Published to TestPyPI!"
    echo "Test install: pip install --index-url https://test.pypi.org/simple/ gac"
elif [ "$choice" = "2" ]; then
    echo "Uploading to PyPI..."
    twine upload dist/*
    echo "✅ Published to PyPI!"
    echo "Install: pip install gac"
else
    echo "❌ Invalid choice"
    exit 1
fi
