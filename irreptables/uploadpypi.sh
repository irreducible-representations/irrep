rm dist/*
rm */irrep.egg-info  */irreptables.egg-info  build
python -m build
python -m twine upload  -r irreptables  dist/*

