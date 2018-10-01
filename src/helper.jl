using Logging
LogLevel(Logging.Info)
using PyCall
"""
    Load a python module from a given absolute path
"""
function pyimport_module(filepath, modulename)
    local imp = PyCall.pywrap(PyCall.pyimport("imp"))
    local spec = imp.util[:spec_from_file_location](modulename, filepath)
    local preloadmod = imp.util[:module_from_spec](spec)
    spec[:loader][:exec_module](preloadmod)
    return preloadmod
end
