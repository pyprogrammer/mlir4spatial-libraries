package tensorflow_lattice.tests

import mlir_libraries.types.TypeImplicits._
import mlir_libraries.{CoprocessorScope, Tensor => MLTensor}
import spatial.dsl._

object PWLCalibrationTest {
  val iterations = 20
  val input_keypoints = MLTensor(values=scala.Array(-1, 0, 3.75, 12), shape=scala.Array(4))
  val pwl_kernel = MLTensor(values = scala.Array(-2.0, 0.46153846, 1.7307693, 3.8076923), shape=scala.Array(4, 1))

  val test_inputs = scala.Array(7.06933614, -4.12573739, 8.87435122, 3.55834566, -4.55233995,
    8.0492217, 1.03970506, -0.04190232, 7.99633263, 9.99942084,
    0.43291209, 0.08522124, 8.09481431, 3.6247634, -3.07901132,
    -2.80682363, 0.86179974, 2.32575243, 6.86811684, -2.89195468).take(iterations)

  val golden = scala.Array(1.7243088, -2.0, 2.557393, 0.10385191, -2.0,
    2.176564, -1.0585977, -1.557801, 2.1521535, 3.0766559,
    -1.338656, -1.4991287, 2.1976066, 0.13450623, -2.0,
    -2.0, -1.1407079, -0.46503723, 1.6314386, -2.0).take(iterations)
}

@spatial class PWLCalibrationTest extends SpatialTest {

  type T = spatial.dsl.FixPt[TRUE, _5, _27]
  val dimensions = I32(1)

  def main(args: Array[String]): Unit = {
    val iterations = I32(PWLCalibrationTest.iterations)
    val input_DRAM = DRAM[T](iterations, dimensions)
    setMem(input_DRAM, Array((PWLCalibrationTest.test_inputs map {
      argon.uconst[T](_)
    }): _*))
    val output_DRAM = DRAM[T](I32(PWLCalibrationTest.iterations))
    Accel {
      val input_sram = SRAM[T](iterations, dimensions)
      input_sram load input_DRAM(I32(0) :: iterations, I32(0) :: dimensions)
      val output_sram = SRAM[T](iterations)
      Pipe {
        CoprocessorScope {
          c =>
            implicit val cps = c
            tensorflow_lattice.tfl.PWLCalibration(pwl_calibration_kernel = PWLCalibrationTest.pwl_kernel, input_keypoints = PWLCalibrationTest.input_keypoints)(input_sram)
        } {
          case (scope, pwl) =>
            val interface = pwl.getInterface
            Pipe {
              Pipe.Foreach(iterations by I32(1)) { i =>
                interface.enq(Seq(i, I32(0)), Set(Bit(true)))
                output_sram(i) = interface.deq(Seq(i, I32(0)), Set(Bit(true)))
              }

              output_DRAM store output_sram
              scope.kill()
            }
        }
      }
    }
    val golden = Array[T]((PWLCalibrationTest.golden map {
      argon.uconst[T](_)
    }): _*)
    val output = getMem(output_DRAM)
    printArray(output, "Output: ")
    (0 until iterations) foreach {
      i =>
        val diff = abs(golden(i) - output(i))
        assert(diff < 1e-3, r"Expected (${golden(i)}), got (${output(i)}) < 1e-3 at index ${i}")
    }
    assert(Bit(true), "Executes")
  }
}
