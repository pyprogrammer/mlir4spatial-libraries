package mlir_libraries.tests

import mlir_libraries.{CoprocessorScope, utils}
import spatial.dsl._
import _root_.spatial.dsl

class TestProcessor(scope: mlir_libraries.CoprocessorScope) extends mlir_libraries.Coprocessor[I32, I32] {
  override def coprocessorScope: CoprocessorScope = scope
  override def deq(inputs: InT, ens: Set[Bit]): OutT = {
    inputs + I32(1)
  }

  override def enq(input: dsl.I32, ens: Set[Bit]): scala.Unit = {}
}

@spatial class CoprocessorTests(threads: scala.Int, workers: scala.Int) extends SpatialTest {
  val test_size = 32

  override def main(args: Array[String]): Void = {

    val output: DRAM2[I32] = DRAM[I32](test_size, test_size)

    Accel {
      val sram: SRAM2[I32] = SRAM[I32](test_size, test_size)
      Pipe {
        mlir_libraries.CoprocessorScope {
          scope =>
            Range(0, workers) map {_ => new TestProcessor(scope)}
        } {
          case (scope, proc) =>
            // split space
            Pipe {
              Stream {
                Range(0, threads) foreach {
                  block_id =>
                    val block_size = test_size / threads
                    val start = block_size * block_id
                    val end = start + block_size
                    val proc_id = block_id % workers
                    val interface = proc(proc_id).interface
                    Foreach(0 until test_size, start until end) {
                      case (i, j) =>
                        interface.enq(i * j)
                    }
                    Foreach(0 until test_size, start until end) {
                      case (i, j) =>
                        sram(i, j) = interface.deq()
                    }
                }
              }
              scope.kill()
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
            assert((i * j + I32(1)) === result(i, j))
        }
    }

    writeCSV2D(result, "result.csv")
  }
}

@spatial class CoprocessorRefs(threads: scala.Int, workers: scala.Int) extends SpatialTest {
  val test_size = 32

  override def main(args: Array[String]): Void = {

    val output: DRAM2[I32] = DRAM[I32](test_size, test_size)

    Accel {
      val sram: SRAM2[I32] = SRAM[I32](test_size, test_size)
      Foreach(0 until test_size, 0 until test_size par I32(workers)) {
        (i, j) =>
          sram(i, j) = i * j + I32(1)
      }
      output store sram
    }

    val result = getMatrix(output)

    writeCSV2D(result, "result.csv")
    (0 until test_size) foreach {
      i =>
        (0 until test_size) foreach {
          j =>
            assert((i * j + I32(1)) === result(i, j))
        }
    }



    assert(Bit(true))
  }
}

class CoprocessorTest1 extends CoprocessorTests(1, 1)
class CoprocessorTest2 extends CoprocessorTests(2, 1)
//class CoprocessorTest4 extends CoprocessorTests(4, 2)  // 1724 cycles
//class CoprocessorTest8 extends CoprocessorTests(8, 2)

class CoprocessorRef1 extends CoprocessorRefs(1, 1)