rm dist/*
rm */irrep.egg-info  */irreptables.egg-info  build
python3 -m build
python3 -m twine upload  -u __token__ dist/*

