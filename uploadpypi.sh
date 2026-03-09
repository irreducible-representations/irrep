rm dist/*
rm */irrep.egg-info  */irreptables.egg-info  build
python -m build
python -m twine upload  -r irrep dist/*

# Add git tag
version="v$(python setup.py --version)"
git tag -a $version -m "release of $version"
git push origin $version
