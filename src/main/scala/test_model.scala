import mlir_libraries.ConversionImplicits._
import tensorflow_lattice.{tf, tfl}
import spatial.libdsl._

object model {
  import mlir_libraries.types._
  def apply[T:Num](arg0_0: Readable2D[T], arg0_1: Readable2D[T], arg0_2: Readable2D[T], arg0_3: Readable2D[T], arg0_4: Readable2D[T])(implicit state:argon.State): (Readable2D[T]) = {
    val (v0) = tfl.CategoricalCalibration(categorical_calibration_kernel = scala.Array(scala.Array(9.643627e-01), scala.Array(9.663681e-01), scala.Array(9.807897e-01), scala.Array(9.702780e-01), scala.Array(9.716194e-01), scala.Array(9.812367e-01), scala.Array(9.716137e-01), scala.Array(9.958263e-01), scala.Array(9.818256e-01), scala.Array(9.634934e-01), scala.Array(9.349014e-01), scala.Array(9.521672e-01), scala.Array(9.782011e-01), scala.Array(9.818071e-01), scala.Array(9.889041e-01), scala.Array(9.056113e-01), scala.Array(9.610040e-01), scala.Array(9.698309e-01), scala.Array(9.983118e-01), scala.Array(9.923077e-01), scala.Array(9.909725e-01), scala.Array(9.200633e-01), scala.Array(8.414015e-01), scala.Array(9.285848e-01), scala.Array(9.955735e-01), scala.Array(9.928998e-01), scala.Array(9.675968e-01), scala.Array(9.156144e-01), scala.Array(8.798674e-01), scala.Array(9.753289e-01), scala.Array(9.805234e-01), scala.Array(9.990155e-01), scala.Array(9.771659e-01), scala.Array(9.765818e-01), scala.Array(8.720487e-01), scala.Array(9.229032e-01), scala.Array(9.941354e-01), scala.Array(9.761051e-01), scala.Array(9.803896e-01), scala.Array(8.433377e-01), scala.Array(7.422317e-01), scala.Array(8.982297e-01), scala.Array(9.875894e-01), scala.Array(9.882098e-01), scala.Array(9.950020e-01), scala.Array(8.758852e-01), scala.Array(7.756904e-01), scala.Array(9.516530e-01), scala.Array(9.964968e-01), scala.Array(9.925848e-01), scala.Array(9.972910e-01), scala.Array(9.550064e-01), scala.Array(7.412706e-01), scala.Array(8.711768e-01)))(arg0_0)
    val (v1) = tfl.PWLCalibration(input_keypoints = scala.Array(4.000000e+01, 1.030000e+02, 1.600000e+02, 2.080000e+02, 2.620000e+02, 3.260000e+02, 4.100000e+02, 8.930000e+02), pwl_calibration_kernel = scala.Array(scala.Array(9.996647e-01), scala.Array(-7.736331e-02), scala.Array(-6.909579e-02), scala.Array(-5.754226e-02), scala.Array(-6.321329e-02), scala.Array(-7.559127e-02), scala.Array(-9.724253e-02), scala.Array(-5.575726e-01)))(arg0_1)
    val (v2) = tfl.PWLCalibration(input_keypoints = scala.Array(1.410000e+02, 2.780000e+02, 3.450000e+02, 4.030000e+02, 4.690000e+02, 5.450000e+02, 6.470000e+02, 1.023000e+03), pwl_calibration_kernel = scala.Array(scala.Array(9.996724e-01), scala.Array(-1.854596e-01), scala.Array(-7.676810e-02), scala.Array(-7.329530e-02), scala.Array(-7.589293e-02), scala.Array(-8.594930e-02), scala.Array(-1.143694e-01), scala.Array(-3.758774e-01)))(arg0_2)
    val (v3) = tfl.PWLCalibration(input_keypoints = scala.Array(0.000000e+00, 1.584336e+00, 3.168673e+00, 4.753009e+00, 6.337346e+00, 7.921682e+00, 9.506019e+00, 1.109035e+01), pwl_calibration_kernel = scala.Array(scala.Array(6.544334e-02), scala.Array(-4.895035e-02), scala.Array(1.713352e-02), scala.Array(5.307740e-03), scala.Array(3.874411e-01), scala.Array(1.723882e-01), scala.Array(5.948114e-02), scala.Array(-2.511748e-01)))(arg0_3)
    val (v4) = tfl.PWLCalibration(input_keypoints = scala.Array(0.000000e+00, 1.046429e+03, 2.092857e+03, 3.139286e+03, 4.185714e+03, 5.232143e+03, 6.278571e+03, 7.325000e+03), pwl_calibration_kernel = scala.Array(scala.Array(2.792139e-01), scala.Array(5.042270e-02), scala.Array(2.910304e-02), scala.Array(1.531714e-02), scala.Array(3.074047e-02), scala.Array(-1.046830e-02), scala.Array(-4.260594e-02), scala.Array(2.361241e-01)))(arg0_4)
    val (v5) = tf.Concatenate(axis = -1)(v0, v1, v2, v3, v4)
    val (v6) = tfl.Lattice(lattice_kernel = scala.Array(scala.Array(1.596640e-01), scala.Array(5.118317e-01), scala.Array(2.848675e-01), scala.Array(3.652959e-01), scala.Array(2.905740e-01), scala.Array(6.220044e-01), scala.Array(6.204429e-01), scala.Array(7.608362e-01), scala.Array(5.591685e-01), scala.Array(8.282099e-01), scala.Array(3.914365e-01), scala.Array(5.561576e-01), scala.Array(9.621527e-01), scala.Array(9.498574e-01), scala.Array(9.849831e-01), scala.Array(9.690263e-01), scala.Array(1.815298e-01), scala.Array(2.278224e-01), scala.Array(3.504758e-01), scala.Array(3.001881e-01), scala.Array(3.880988e-04), scala.Array(3.807952e-04), scala.Array(8.134229e-01), scala.Array(8.468219e-01), scala.Array(8.589996e-01), scala.Array(9.742905e-01), scala.Array(3.040237e-01), scala.Array(3.735402e-01), scala.Array(9.362645e-01), scala.Array(9.282511e-01), scala.Array(9.924486e-01), scala.Array(9.907513e-01)), tp = "hypercube", shape = scala.Array(2, 2, 2, 2, 2), units = 1)(v5)
    val (v7) = tfl.PWLCalibration(input_keypoints = scala.Array(0.000000e+00, 1.428571e-01, 2.857143e-01, 4.285714e-01, 5.714286e-01, 7.142857e-01, 8.571429e-01, 1.000000e+00), pwl_calibration_kernel = scala.Array(scala.Array(-5.790979e+03), scala.Array(1.123730e+03), scala.Array(1.123423e+03), scala.Array(1.123504e+03), scala.Array(1.122725e+03), scala.Array(1.122944e+03), scala.Array(1.125060e+03), scala.Array(1.126378e+03)))(v6)
    return (v7)
  }
}

@spatial object App extends SpatialApp {
  import spatial.dsl._
  type T = spatial.dsl.FixPt[TRUE, _24, _8]
  val dimensions = 5
  val iterations = 8
  def main(args: Array[String]) : Unit = {
//    val input = loadCSV1D[T](s"${System.getProperty("user.dir")}/test_parameters/simplex/5-2/input.csv")

    val input_DRAM = DRAM[T](iterations, dimensions)

//    setMem(input_DRAM, input)

//    val iterations = ArgIn[Int]
//    setArg(iterations, args(0).to[Int])

    val output_DRAM = DRAM[T](iterations)

    Accel {
      val input_sram = SRAM[T](iterations, dimensions)
      input_sram load input_DRAM(0 :: iterations, 0 :: dimensions)

      val output_sram = SRAM[T](iterations)
      val result = model(input_sram, input_sram, input_sram, input_sram, input_sram)
      Pipe.Foreach(iterations by 1) { i =>
        output_sram(i) = result(i, 0)
      }

      output_DRAM store output_sram
    }
    println(getMem(output_DRAM))
  }
}
