package mlir_libraries.tests

import mlir_libraries.{CoprocessorScope, utils}
import spatial.dsl._

class TestProcessor(scope: mlir_libraries.CoprocessorScope) extends mlir_libraries.Coprocessor[I32, I32](2, 1, 1) {
  override def coprocessorScope: CoprocessorScope = scope
  override def execute(inputs: Seq[I32]): Seq[I32] = {
    Seq((inputs reduceTree { _ + _ }) + I32(1))
  }
}

@spatial class CoprocessorTests(par_factor: scala.Int) extends SpatialTest {
  val test_size = 16

  override def main(args: Array[String]): Void = {

    val output: DRAM2[I32] = DRAM[I32](test_size, test_size)

    Accel {
      val sram: SRAM2[I32] = SRAM[I32](test_size, test_size)
      Pipe {
        mlir_libraries.CoprocessorScope {
          scope =>
            new TestProcessor(scope)
        } {
          proc =>
            Foreach(0 until test_size, 0 until test_size par I32(par_factor)) {
              case (i, j) =>
                sram(i, j) = proc(Seq(i, j)).head
            }
        }
        output store sram
      }
    }

    val result = getMatrix(output)
    (0 until test_size) foreach {
      i =>
        (0 until test_size) foreach {
          j =>
            assert((i + j + I32(1)) === result(i, j))
        }
    }

    writeCSV2D(result, "result.csv")

    assert(Bit(true))
  }
}

class CoprocessorTest1 extends CoprocessorTests(1)
