from driverConstants import *
from driverExplicitMPI import ExplicitMPIAnalysis
import driverUtils, sys
options = {
    'SIMExt':'.sim',
    'ams':OFF,
    'analysisType':EXPLICIT,
    'applicationName':'analysis',
    'aqua':OFF,
    'ask_delete':OFF,
    'beamSectGen':OFF,
    'biorid':OFF,
    'cavityTypes':[],
    'cavparallel':OFF,
    'complexFrequency':OFF,
    'contact':ON,
    'cosimulation':OFF,
    'coupledProcedure':OFF,
    'cpus':4,
    'cse':OFF,
    'cyclicSymmetryModel':OFF,
    'directCyclic':OFF,
    'domains':4,
    'dsa':OFF,
    'dynamic':OFF,
    'filPrt':[],
    'fils':[],
    'finitesliding':ON,
    'foundation':OFF,
    'geostatic':OFF,
    'geotech':OFF,
    'heatTransfer':OFF,
    'importer':OFF,
    'importerParts':OFF,
    'includes':[],
    'initialConditionsFile':OFF,
    'input':'Running_INP',
    'inputFormat':INP,
    'interactive':None,
    'job':'Running_INP',
    'keyword_licenses':[],
    'lanczos':OFF,
    'libs':[],
    'magnetostatic':OFF,
    'massDiffusion':OFF,
    'modifiedTet':OFF,
    'moldflowFiles':[],
    'moldflowMaterial':OFF,
    'mp_file_system':(DETECT, DETECT),
    'mp_head_node':('obg-304793', '192.168.1.34', 'obg-304793.adsroot.itcs.umich.edu', '35.7.4.180'),
    'mp_host_list':(('obg-304793', 4),),
    'mp_mode':MPI,
    'mp_mpi_validate':OFF,
    'mp_mpirun_path':'C:\\Program Files\\Microsoft MPI\\bin\\mpiexec.exe',
    'mp_rsh_command':'dummy %H -l DeLancey -n %C',
    'multiphysics':OFF,
    'noDmpDirect':[],
    'noMultiHost':[],
    'noMultiHostElemLoop':[],
    'noStdParallel':[],
    'no_domain_check':1,
    'outputKeywords':ON,
    'parallel':DOMAIN,
    'parallel_odb':SINGLE,
    'parallel_restartfile':SINGLE,
    'parameterized':OFF,
    'partsAndAssemblies':ON,
    'parval':OFF,
    'postOutput':OFF,
    'preDecomposition':OFF,
    'restart':OFF,
    'restartEndStep':OFF,
    'restartIncrement':0,
    'restartStep':0,
    'restartWrite':ON,
    'rezone':OFF,
    'runCalculator':OFF,
    'soils':OFF,
    'soliter':OFF,
    'solverTypes':['DIRECT'],
    'staticNonlinear':OFF,
    'steadyStateTransport':OFF,
    'step':ON,
    'subGen':OFF,
    'subGenLibs':[],
    'subGenTypes':[],
    'submodel':OFF,
    'substrLibDefs':OFF,
    'substructure':OFF,
    'symmetricModelGeneration':OFF,
    'thermal':OFF,
    'tmpdir':'C:\\Users\\DELANC~1\\AppData\\Local\\Temp',
    'tracer':OFF,
    'visco':OFF,
}
analysis = ExplicitMPIAnalysis(options)
status = analysis.run()
sys.exit(status)
