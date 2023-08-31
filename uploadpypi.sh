rm dist/*
rm */irrep.egg-info  */irreptables.egg-info  build
python3 -m build
python3 -m twine upload  -u __token__ dist/*

# Add git tag
version="v$(python3 setup.py --version)"
git tag -a $version -m "release of $version"
git push origin $version
