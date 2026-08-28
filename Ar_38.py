from qiskit.quantum_info import SparsePauliOp
import hamiltonian as hamm
import TreeTensorNetworks as ttn

formattedH = (-93.9749221802 * SparsePauliOp.from_list([('III', 1)]) +
              +0.1799961627 * SparsePauliOp.from_list([('XII', 1)]) +
              -56.0362358093 * SparsePauliOp.from_list([('ZII', 1)]) +
              +0.2252326235 * SparsePauliOp.from_list([('IXI', 1)]) +
              +19.2602767944 * SparsePauliOp.from_list([('IZI', 1)]) +
              -0.1433962137 * SparsePauliOp.from_list([('XXI', 1)]) +
              +0.2252326235 * SparsePauliOp.from_list([('ZXI', 1)]) +
              -0.1799961627 * SparsePauliOp.from_list([('XZI', 1)]) +
              +0.1433962137 * SparsePauliOp.from_list([('YYI', 1)]) +
              -18.6784095764 * SparsePauliOp.from_list([('ZZI', 1)]) +
              -0.0349043496 * SparsePauliOp.from_list([('IIX', 1)]) +
              -19.2592506409 * SparsePauliOp.from_list([('IIZ', 1)]) +
              -0.0349043533 * SparsePauliOp.from_list([('XIX', 1)]) +
              -0.0349043496 * SparsePauliOp.from_list([('ZIX', 1)]) +
              +0.1799961627 * SparsePauliOp.from_list([('XIZ', 1)]) +
              -0.0349043533 * SparsePauliOp.from_list([('YIY', 1)]) +
              +18.6794357300 * SparsePauliOp.from_list([('ZIZ', 1)]) +
              +0.0349043459 * SparsePauliOp.from_list([('IXX', 1)]) +
              +0.0349043570 * SparsePauliOp.from_list([('IZX', 1)]) +
              +0.0615598038 * SparsePauliOp.from_list([('IXZ', 1)]) +
              +0.0349043533 * SparsePauliOp.from_list([('IYY', 1)]) +
              +19.2602806091 * SparsePauliOp.from_list([('IZZ', 1)]) +
              +0.0349043533 * SparsePauliOp.from_list([('XXX', 1)]) +
              +0.0349043459 * SparsePauliOp.from_list([('ZXX', 1)]) +
              +0.0349043533 * SparsePauliOp.from_list([('XZX', 1)]) +
              -0.0349043533 * SparsePauliOp.from_list([('YYX', 1)]) +
              +0.0349043570 * SparsePauliOp.from_list([('ZZX', 1)]) +
              -0.1433962137 * SparsePauliOp.from_list([('XXZ', 1)]) +
              +0.0349043533 * SparsePauliOp.from_list([('YXY', 1)]) +
              +0.0615598038 * SparsePauliOp.from_list([('ZXZ', 1)]) +
              +0.0349043533 * SparsePauliOp.from_list([('XYY', 1)]) +
              -0.1799961627 * SparsePauliOp.from_list([('XZZ', 1)]) +
              +0.0349043533 * SparsePauliOp.from_list([('ZYY', 1)]) +
              +0.0349043533 * SparsePauliOp.from_list([('YZY', 1)]) +
              +0.1433962137 * SparsePauliOp.from_list([('YYZ', 1)]) +
              -18.6784057617 * SparsePauliOp.from_list([('ZZZ', 1)]))

ttn_hamiltonian = hamm.Hamiltonian.from_SparsePauliOp(formattedH)
my_ttn = ttn.ttn(layers = 2, max_bond_dim = 5, local_dim = 2, num_child_nodes = 2, num_roots = 1)
