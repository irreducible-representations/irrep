#rm dist/*
#rm */irrep.egg-info  */irreptables.egg-info  build
#python3 setup.py bdist_wheel
#python3 setup_tables.py bdist_wheel
#rm */irrep.egg-info  */irreptables.egg-info  build

python3 -m twine upload  -u stepan-tsirkin dist/*
