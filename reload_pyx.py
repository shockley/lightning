import distutils.core as core
import os,sys
from Cython.Build import cythonize

# should call
#    sudo pip2.7 install ./ --force-reinstall --upgrade
#after running this script

if len(sys.argv) < 2:
	module = 'karma_sgd_fast'
else:
	module = sys.argv[1]

if os.path.isfile("./lightning/impl/"+module+".cpp"):
	os.remove("./lightning/impl/"+module+".cpp")
core.setup(ext_modules=cythonize("./lightning/impl/"+module+".pyx",language="c++"))
