package mlir_libraries.tests

import mlir_libraries.{CoprocessorScope, utils}
import spatial.dsl._

class TestProcessor(scope: mlir_libraries.CoprocessorScope, prealloc: scala.Int) extends mlir_libraries.Coprocessor[I32, I32](2, 1, prealloc) {
  override def coprocessorScope: CoprocessorScope = scope
  override def execute(inputs: Seq[I32]): Seq[I32] = {
    Seq((inputs reduceTree { _ + _ }) + I32(1))
  }
}

@spatial class CoprocessorTests(threads: scala.Int, workers: scala.Int) extends SpatialTest {
  val test_size = 16

  override def main(args: Array[String]): Void = {

    val output: DRAM2[I32] = DRAM[I32](test_size, test_size)

    Accel {
      val sram: SRAM2[I32] = SRAM[I32](test_size, test_size)
      Pipe {
        mlir_libraries.CoprocessorScope {
          scope =>
            Range(0, workers) map {_ => new TestProcessor(scope, threads / workers)}
        } {
          proc =>
            // split space
            Range(0, threads) foreach {
              block_id =>
                val block_size = test_size / threads
                val start = block_size * block_id
                val end = start + block_size
                val proc_id = block_id % workers
                Foreach(0 until test_size, start until end) {
                  case (i, j) =>
                    sram(i, j) = proc(proc_id)(Seq(i, j)).head
                }
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

class CoprocessorTest1 extends CoprocessorTests(1, 1)
class CoprocessorTest4 extends CoprocessorTests(4, 2)
