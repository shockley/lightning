import distutils.core as core
import os,sys
from Cython.Build import cythonize

if len(sys.argv) != 2:
	print 'error: wrong argument length'+str(len(sys.argv))
	print sys.argv
	quit()
#like karma_sgd_fast, sdca_fast, dual_cd_fast
module = sys.argv[1]
if os.path.isfile("./lightning/impl/"+module+".cpp"):
	os.remove("./lightning/impl/"+module+".cpp")
core.setup(ext_modules=cythonize("./lightning/impl/"+module+".pyx",language="c++"))
