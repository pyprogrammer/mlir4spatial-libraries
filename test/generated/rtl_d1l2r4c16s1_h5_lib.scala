package generated
import spatial.libdsl._
import tensorflow_lattice._
import mlir_libraries.ConversionImplicits._
object rtl_d1l2r4c16s1_h5_callable {
  def rtl_d1l2r4c16s1_h5_callable[F32:Num](arg0_0: mlir_libraries.types.ReadableND[F32], arg0_1: mlir_libraries.types.ReadableND[F32], arg0_2: mlir_libraries.types.ReadableND[F32], arg0_3: mlir_libraries.types.ReadableND[F32], arg0_4: mlir_libraries.types.ReadableND[F32], arg0_5: mlir_libraries.types.ReadableND[F32], arg0_6: mlir_libraries.types.ReadableND[F32], arg0_7: mlir_libraries.types.ReadableND[F32], arg0_8: mlir_libraries.types.ReadableND[F32])(implicit state: argon.State, config: mlir_libraries.OptimizationConfig): (mlir_libraries.types.ReadableND[F32]) = {
    import mlir_libraries._
    val (v0) = tfl.PWLCalibration(input_keypoints = Tensor(values=Array(0.000000e+00, 2.540000e+02, 6.420000e+02, 1.326000e+03, 2.400000e+03, 3.425000e+03, 4.629000e+03, 5.800000e+03, 7.979000e+03, 9.838000e+03, 1.243100e+04, 1.515200e+04, 2.156100e+04, 3.027700e+04, 3.834200e+04, 4.290800e+04), shape = Array(16)), pwl_calibration_kernel = Tensor(values=Array(8.926245e-06, 4.975775e-06, 7.717421e-03, 1.495382e-02, 2.208480e-02, 1.800093e-02, 2.115256e-02, 2.050287e-02, 4.369846e-02, 3.636475e-02, 5.358467e-02, 5.777515e-02, 1.448128e-01, 2.021511e-01, 1.879334e-01, 1.046283e-01), shape = Array(16, 1)))(arg0_0)
    val (v1) = tfl.CategoricalCalibration(categorical_calibration_kernel = Tensor(values=Array(4.062594e-01, 5.166234e-01, 5.748255e-01, 5.000000e-01), shape = Array(4, 1)))(arg0_1)
    val (v2) = tfl.CategoricalCalibration(categorical_calibration_kernel = Tensor(values=Array(5.021144e-01, 5.004148e-01, 4.958543e-01, 4.999891e-01, 4.996600e-01, 4.965046e-01, 4.964256e-01, 4.973226e-01, 4.970912e-01, 4.973097e-01, 4.970236e-01, 4.978300e-01, 5.805076e-01, 4.975426e-01, 4.390954e-01, 4.471659e-01, 4.976497e-01, 4.973326e-01, 5.014743e-01, 5.019920e-01, 5.260766e-01, 4.971820e-01, 4.999948e-01, 4.976301e-01, 6.528574e-01, 4.999895e-01, 4.972339e-01, 4.999844e-01, 4.967490e-01, 4.966083e-01, 4.979359e-01, 4.984089e-01, 4.978267e-01, 4.976190e-01, 4.976569e-01, 4.977028e-01, 4.976136e-01, 4.979578e-01, 4.982244e-01, 4.981339e-01, 4.982791e-01, 4.969158e-01, 4.984170e-01, 5.020069e-01, 5.014656e-01, 4.998622e-01, 4.995237e-01, 5.019811e-01, 4.995571e-01, 3.946873e-01, 5.001258e-01, 4.995602e-01, 4.994966e-01, 4.996493e-01, 5.451884e-01, 4.985943e-01, 4.984490e-01, 4.978717e-01, 4.973731e-01, 4.976518e-01, 5.014415e-01, 5.000317e-01, 4.999856e-01, 4.977189e-01, 5.001473e-01, 5.084401e-01, 4.956169e-01, 4.967147e-01, 4.970143e-01, 4.965473e-01, 5.000000e-01), shape = Array(71, 1)))(arg0_2)
    val (v3) = tfl.CategoricalCalibration(categorical_calibration_kernel = Tensor(values=Array(4.998352e-01, 4.826059e-01, 4.879639e-01, 4.991390e-01, 4.780235e-01, 3.337203e-01, 5.060279e-01, 5.017694e-01, 5.007132e-01, 7.188101e-01, 4.977915e-01, 5.000000e-01), shape = Array(12, 1)))(arg0_3)
    val (v4) = tfl.PWLCalibration(input_keypoints = Tensor(values=Array(0.000000e+00, 2.260000e+02, 4.520000e+02, 6.950000e+02, 9.180000e+02, 1.140000e+03, 1.363000e+03, 1.589000e+03, 1.837000e+03, 2.127000e+03, 2.489000e+03, 2.918000e+03, 3.940000e+03, 1.040600e+04, 1.537700e+04, 1.379964e+09), shape = Array(16)), pwl_calibration_kernel = Tensor(values=Array(8.410276e-06, 8.410273e-06, 7.831357e-02, 7.831352e-02, 1.928481e-02, 1.928479e-02, 1.131334e-02, 1.131278e-02, 4.528888e-03, 4.529186e-03, 6.418504e-03, 6.418444e-03, 1.254742e-02, 1.254806e-02, 9.716019e-03, 9.715840e-03, 7.744640e-03, 7.744819e-03, 6.242037e-03, 6.241918e-03, 5.335927e-03, 5.336136e-03, 4.734665e-03, 4.734457e-03, 3.932118e-03, 3.931969e-03, 2.075821e-03, 2.075911e-03, 3.127456e-04, 3.127158e-04, 8.184490e-01, 8.184490e-01), shape = Array(16, 2)))(arg0_4)
    val (v5) = tfl.PWLCalibration(input_keypoints = Tensor(values=Array(0.000000e+00, 6.390000e+02, 1.269000e+03, 1.912000e+03, 2.591000e+03, 3.366000e+03, 4.265000e+03, 5.357000e+03, 6.590000e+03, 8.116000e+03, 9.849000e+03, 1.185000e+04, 1.505000e+04, 2.050400e+04, 3.305600e+04, 1.309937e+09), shape = Array(16)), pwl_calibration_kernel = Tensor(values=Array(8.634500e-06, 6.812286e-02, 3.590708e-02, 2.607918e-02, 2.010302e-02, 1.591294e-02, 1.270109e-02, 1.032947e-02, 8.517310e-03, 7.135049e-03, 6.804675e-03, 5.414516e-03, 4.033834e-03, 2.710596e-03, 1.491517e-03, 7.747202e-01), shape = Array(16, 1)))(arg0_5)
    val (v6) = tfl.CategoricalCalibration(categorical_calibration_kernel = Tensor(values=Array(4.988178e-01, 4.999921e-01, 5.000000e-01), shape = Array(3, 1)))(arg0_6)
    val (v7) = tfl.PWLCalibration(input_keypoints = Tensor(values=Array(0.000000e+00, 1.000000e+00, 3.000000e+00), shape = Array(3)), pwl_calibration_kernel = Tensor(values=Array(9.005475e-06, 3.204392e-01, 6.654968e-01), shape = Array(3, 1)))(arg0_7)
    val (v8) = tfl.PWLCalibration(input_keypoints = Tensor(values=Array(0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00), shape = Array(4)), pwl_calibration_kernel = Tensor(values=Array(8.159171e-06, 3.305864e-01, 3.332107e-01, 3.331752e-01), shape = Array(4, 1)))(arg0_8)
    val (v9) = tf.Concatenate(axis = 1)(v0, v1, v2, v3, v4, v5, v6, v7, v8)
    val (v10_) = tf.GatherV2(axis = 1, indices = Tensor(values=Array(8, 1, 3, 2, 0, 7, 4, 6, 9, 5), shape = Array(2, 5)))(v9)
    val v10 = mlir_libraries.spatiallib.Materialize()(v10_)
    val (v11) = tfl.Lattice(lattice_kernel = Tensor(values=Array(0.000000e+00, 1.293323e-05, 1.977806e-01, 2.076020e-01, 1.776780e-01, 2.000015e-01, 3.983018e-01, 4.000000e-01, 1.952122e-01, 2.082492e-01, 3.982245e-01, 3.999954e-01, 4.620508e-01, 4.000000e-01, 5.985350e-01, 6.000000e-01, 1.428304e-01, 2.076021e-01, 3.979124e-01, 4.001754e-01, 3.893842e-01, 4.000000e-01, 5.983977e-01, 6.000000e-01, 4.199177e-01, 3.999954e-01, 5.982192e-01, 6.000000e-01, 6.856353e-01, 6.000000e-01, 7.985759e-01, 8.000000e-01, 1.994204e-01, 1.688023e-01, 4.000000e-01, 4.073152e-01, 3.999185e-01, 4.000011e-01, 6.000000e-01, 6.000000e-01, 3.993237e-01, 4.079354e-01, 6.000000e-01, 5.999590e-01, 5.999094e-01, 6.000000e-01, 8.000000e-01, 8.000000e-01, 3.998109e-01, 4.073152e-01, 6.000000e-01, 5.999470e-01, 5.999892e-01, 6.000000e-01, 8.000000e-01, 8.000000e-01, 5.998634e-01, 5.999590e-01, 8.000000e-01, 8.000000e-01, 7.998885e-01, 8.000000e-01, 1.000000e+00, 1.000000e+00), shape = Array(32, 2)), shape = Tensor(values=Array(2, 2, 2, 2, 2), shape = Array(5)), tp = "hypercube", units = 2)(v10)
    val (v12) = tf.Concatenate(axis = 1)(v11)
    val (v13) = tfl.Linear(linear_layer_bias = 4.110190e-02, linear_layer_kernel = Tensor(values=Array(3.455284e-01, 1.458920e-01), shape = Array(2, 1)))(v12)
    val (v14) = tfl.PWLCalibration(input_keypoints = Tensor(values=Array(0.000000e+00, 3.225806e-02, 6.451613e-02, 9.677419e-02, 1.290323e-01, 1.612903e-01, 1.935484e-01, 2.258064e-01, 2.580645e-01, 2.903226e-01, 3.225806e-01, 3.548387e-01, 3.870968e-01, 4.193548e-01, 4.516129e-01, 4.838710e-01, 5.161290e-01, 5.483871e-01, 5.806451e-01, 6.129032e-01, 6.451613e-01, 6.774194e-01, 7.096774e-01, 7.419355e-01, 7.741935e-01, 8.064516e-01, 8.387097e-01, 8.709677e-01, 9.032258e-01, 9.354839e-01, 9.677419e-01, 1.000000e+00), shape = Array(32)), pwl_calibration_kernel = Tensor(values=Array(0.000000e+00, 6.431583e-02, 5.744133e-02, 3.126496e-02, 2.715141e-03, 6.273828e-02, 6.323135e-01, 5.630970e-03, 1.013279e-06, 7.301986e-03, 7.302284e-03, 7.301152e-03, 7.300735e-03, 7.301748e-03, 7.302880e-03, 7.302165e-03, 7.302642e-03, 7.302403e-03, 7.302284e-03, 7.302284e-03, 7.302284e-03, 7.302284e-03, 7.302284e-03, 7.302284e-03, 7.302284e-03, 7.302284e-03, 7.302284e-03, 7.302284e-03, 4.834116e-03, 1.966953e-06, 1.311302e-06, 7.152557e-07), shape = Array(32, 1)))(v13)
    val (v15) = tf.Minimum(constant = 1.000000e+00)(v14)
    val (v16) = tf.Maximum(constant = 0.000000e+00)(v15)
    return (v16)
  }
}
