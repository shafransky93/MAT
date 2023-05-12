from qiskit import IBMQ
from qiskit.utils import QuantumInstance
from qiskit.algorithms import Shor

IBMQ.enable_account('b19450f666ada166dceba9bc938493b4ae6ae3253d0efda7e48e7ef3852eec6854cd822bd824169c30189154ba7b7d7f10d857c37259a7dcb9341415ac1ad645') # Enter your API token here
provider = IBMQ.get_provider(hub='ibm-q')

backend = provider.get_backend('ibmq_qasm_simulator') # Specifies the quantum device

print('\n Shors Algorithm')
print('--------------------')
print('\nExecuting...\n')

factors = Shor(QuantumInstance(backend, shots=100, skip_qobj_validation=False)) 

inp = input('Value to be factorized:')
result_dict = factors.factor(N=int(inp), a=2) # Where N is the integer to be factored
result = result_dict.factors

print(result)
print('\nPress any key to close')
input()
