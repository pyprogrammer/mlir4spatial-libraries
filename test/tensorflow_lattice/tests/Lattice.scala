package tensorflow_lattice.tests

import spatial.dsl._
import mlir_libraries.ConversionImplicits._

object LatticeTest {
  val iterations = 20
  val lattice_kernel = scala.Array(0.0, 0.2, 0.2, 0.4, 0.2, 0.4, 0.4, 0.6, 0.2,
    0.4, 0.4, 0.6, 0.4, 0.6, 0.6, 0.8, 0.2, 0.4, 0.4, 0.6, 0.4, 0.6, 0.6, 0.8,
    0.4, 0.6, 0.6, 0.8, 0.6, 0.8, 0.8, 1.0) map { x => scala.Array[scala.Double](x) }

  val test_inputs = scala.Array(
    scala.Array(0.35814493, 0.45876343, 0.26052009, 0.4633784, 0.59289284),
    scala.Array(0.8157901, 0.13248052, 0.93235214, 0.3562312, 0.72890972),
    scala.Array(0.52692072, 0.49771845, 0.7084408, 0.20148453, 0.7751054),
    scala.Array(0.9887156, 0.0299881, 0.85561108, 0.74709715, 0.50709276),
    scala.Array(0.07330725, 0.37191703, 0.67804784, 0.95678098, 0.53233605),
    scala.Array(0.19533886, 0.7076565, 0.85398532, 0.48035049, 0.94513752),
    scala.Array(0.14004581, 0.6393042, 0.93354944, 0.44365684, 0.70958663),
    scala.Array(0.09698129, 0.43531014, 0.67196545, 0.46226384, 0.8852037),
    scala.Array(0.26349345, 0.11072366, 0.60795466, 0.21402168, 0.26159861),
    scala.Array(0.89511044, 0.79520253, 0.64677987, 0.49362013, 0.27954056),
    scala.Array(0.91319689, 0.45773806, 0.06550992, 0.56996202, 0.71523172),
    scala.Array(0.62480322, 0.72595559, 0.66580501, 0.6147201, 0.45875612),
    scala.Array(0.70786419, 0.49884259, 0.5803606, 0.68740088, 0.60213667),
    scala.Array(0.10587894, 0.0270324, 0.25930853, 0.42129714, 0.1135602),
    scala.Array(0.26970441, 0.90450072, 0.75622349, 0.91636219, 0.2066733),
    scala.Array(0.385794, 0.8872154, 0.1511485, 0.60721121, 0.8119909),
    scala.Array(0.47223358, 0.74109598, 0.60515997, 0.49301198, 0.39645178),
    scala.Array(0.1641796, 0.68478382, 0.14134886, 0.86807823, 0.71831843),
    scala.Array(0.39605967, 0.00730522, 0.97777832, 0.92250456, 0.42590122),
    scala.Array(0.86277249, 0.85204319, 0.41044324, 0.71383252, 0.2207585)
  ).take(iterations).flatten

  val golden = scala.Array(
    0.42673993,
    0.59315276,
    0.541934,
    0.62570095,
    0.52247787,
    0.6364937,
    0.5732286,
    0.5103449,
    0.2915584,
    0.62205076,
    0.54432774,
    0.618008,
    0.615321,
    0.18541543,
    0.61069286,
    0.56867206,
    0.5415907,
    0.51534176,
    0.5459098,
    0.61196995).take(iterations)
}

@spatial class LatticeTest extends SpatialTest {

  type T = spatial.dsl.FixPt[TRUE, _2, _30]
  val dimensions = 5

  def main(args: Array[String]): Unit = {
    val iterations = LatticeTest.iterations
    val input_DRAM = DRAM[T](iterations, dimensions)
    setMem(input_DRAM, Array((LatticeTest.test_inputs map { argon.uconst[T](_) }):_*))
    val output_DRAM = DRAM[T](I32(iterations))

    Accel {
      val input_sram = SRAM[T](iterations, dimensions)
      input_sram load input_DRAM(0 :: iterations, 0 :: dimensions)
      val output_sram = SRAM[T](iterations)
      Pipe.Foreach(iterations by 1) { i =>
        val lattice =
          tensorflow_lattice.tfl.Lattice(tp = "hypercube",
            shape = scala.Array(2, 2, 2, 2, 2),
            units = 1,
            lattice_kernel = LatticeTest.lattice_kernel)(RR2(input_sram))
        output_sram(i) = lattice(i, I32(0))
      }

      output_DRAM store output_sram
    }
    val golden = Array[T]((LatticeTest.golden map { argon.uconst[T](_) }):_*)
    val output = getMem(output_DRAM)
    printArray(output, "Output: ")
    (0 until iterations) foreach {
      i =>
        val diff = abs(golden(i) - output(i))
        assert(diff < 1e-3, r"Expected (${golden(i)}), got (${output(i)}) < 1e-3")
    }
  }
}
