package generated
import spatial.libdsl._
import mlir_libraries.types.TypeImplicits._
import mlir_libraries.{DumpScope, LatticeConfig, LatticeOptions}
@spatial class rtl_d1l2r4c16s1_log2_coproc(repeat_iterations: Int, run_iterations: Int, implicit val config: mlir_libraries.LatticeConfig) extends SpatialTest {

  override def compileArgs = s"--max_cycles=${repeat_iterations*run_iterations * 1000 + 10000} --vv"

  import spatial.dsl._
  type T = FixPt[TRUE, _9, _23]
  val iterations = run_iterations
  val offset = I32(0)
  def main(args: Array[String]) : Unit = {
    val dram_0 = DRAM[T](I32(iterations), I32(1))
    val dram_1 = DRAM[T](I32(iterations), I32(1))
    val dram_2 = DRAM[T](I32(iterations), I32(1))
    val dram_3 = DRAM[T](I32(iterations), I32(1))
    val dram_4 = DRAM[T](I32(iterations), I32(1))
    val dram_5 = DRAM[T](I32(iterations), I32(1))
    val dram_6 = DRAM[T](I32(iterations), I32(1))
    val dram_7 = DRAM[T](I32(iterations), I32(1))
    val dram_8 = DRAM[T](I32(iterations), I32(1))

    val input_0 = loadCSV2D[T]("/local/ssd/home/stanfurd/local-remote-deploy/mlir4spatial-libraries/test/generated/log2d1/input_0.csv")
    val sliced_input_0 = Matrix.tabulate(I32(iterations), I32(1)) {case (row, col) => input_0(row + offset, col)}
    setMem(dram_0, sliced_input_0)
    val input_1 = loadCSV2D[T]("/local/ssd/home/stanfurd/local-remote-deploy/mlir4spatial-libraries/test/generated/log2d1/input_1.csv")
    val sliced_input_1 = Matrix.tabulate(I32(iterations), I32(1)) {case (row, col) => input_1(row + offset, col)}
    setMem(dram_1, sliced_input_1)
    val input_2 = loadCSV2D[T]("/local/ssd/home/stanfurd/local-remote-deploy/mlir4spatial-libraries/test/generated/log2d1/input_2.csv")
    val sliced_input_2 = Matrix.tabulate(I32(iterations), I32(1)) {case (row, col) => input_2(row + offset, col)}
    setMem(dram_2, sliced_input_2)
    val input_3 = loadCSV2D[T]("/local/ssd/home/stanfurd/local-remote-deploy/mlir4spatial-libraries/test/generated/log2d1/input_3.csv")
    val sliced_input_3 = Matrix.tabulate(I32(iterations), I32(1)) {case (row, col) => input_3(row + offset, col)}
    setMem(dram_3, sliced_input_3)
    val input_4 = loadCSV2D[T]("/local/ssd/home/stanfurd/local-remote-deploy/mlir4spatial-libraries/test/generated/log2d1/input_4.csv")
    val sliced_input_4 = Matrix.tabulate(I32(iterations), I32(1)) {case (row, col) => input_4(row + offset, col)}
    setMem(dram_4, sliced_input_4)
    val input_5 = loadCSV2D[T]("/local/ssd/home/stanfurd/local-remote-deploy/mlir4spatial-libraries/test/generated/log2d1/input_5.csv")
    val sliced_input_5 = Matrix.tabulate(I32(iterations), I32(1)) {case (row, col) => input_5(row + offset, col)}
    setMem(dram_5, sliced_input_5)
    val input_6 = loadCSV2D[T]("/local/ssd/home/stanfurd/local-remote-deploy/mlir4spatial-libraries/test/generated/log2d1/input_6.csv")
    val sliced_input_6 = Matrix.tabulate(I32(iterations), I32(1)) {case (row, col) => input_6(row + offset, col)}
    setMem(dram_6, sliced_input_6)
    val input_7 = loadCSV2D[T]("/local/ssd/home/stanfurd/local-remote-deploy/mlir4spatial-libraries/test/generated/log2d1/input_7.csv")
    val sliced_input_7 = Matrix.tabulate(I32(iterations), I32(1)) {case (row, col) => input_7(row + offset, col)}
    setMem(dram_7, sliced_input_7)
    val input_8 = loadCSV2D[T]("/local/ssd/home/stanfurd/local-remote-deploy/mlir4spatial-libraries/test/generated/log2d1/input_8.csv")
    val sliced_input_8 = Matrix.tabulate(I32(iterations), I32(1)) {case (row, col) => input_8(row + offset, col)}
    setMem(dram_8, sliced_input_8)

    val output_DRAM = DRAM[T](I32(iterations))

    implicit val dumpScope = new DumpScope()
    Accel {
      val sram_0 = SRAM[T](I32(iterations), I32(1))
      val sram_1 = SRAM[T](I32(iterations), I32(1))
      val sram_2 = SRAM[T](I32(iterations), I32(1))
      val sram_3 = SRAM[T](I32(iterations), I32(1))
      val sram_4 = SRAM[T](I32(iterations), I32(1))
      val sram_5 = SRAM[T](I32(iterations), I32(1))
      val sram_6 = SRAM[T](I32(iterations), I32(1))
      val sram_7 = SRAM[T](I32(iterations), I32(1))
      val sram_8 = SRAM[T](I32(iterations), I32(1))
      Parallel {
        sram_0 load dram_0
        sram_1 load dram_1
        sram_2 load dram_2
        sram_3 load dram_3
        sram_4 load dram_4
        sram_5 load dram_5
        sram_6 load dram_6
        sram_7 load dram_7
        sram_8 load dram_8
      }

      Pipe {
        val output_sram = SRAM[T](I32(iterations))
        dumpScope.setSramScope
        mlir_libraries.CoprocessorScope {
          scope =>
            implicit val cps = scope
            rtl_d1l2r4c16s1_log2_callable.rtl_d1l2r4c16s1_log2_callable(sram_0, sram_1, sram_2, sram_3, sram_4, sram_5, sram_6, sram_7, sram_8)
          } {
          case (scope, tmp) =>
            val result = tmp.getInterface
            'Enqueue.Pipe.Foreach(I32(repeat_iterations) by I32(1), I32(iterations) by I32(1)) {
              (x, i) =>
                result.enq(Seq(i, I32(0)), Set(Bit(true)))
            }

            Sequential {
              'Dequeue.Pipe.Foreach(I32(repeat_iterations) by I32(1), I32(iterations) by I32(1)) {
                (x, i) =>
                  output_sram(i) = result.deq(Seq(i, I32(0)), Set(Bit(true)))
              }
              Parallel {
                dumpScope.store()
              }
              retimeGate()
              scope.kill()
            }
        }
        output_DRAM store output_sram
      }
    }

    dumpScope.dump()

    if (mlir_libraries.Options.Verify) {
      val golden = loadCSV1D[T]("/local/ssd/home/stanfurd/local-remote-deploy/mlir4spatial-libraries/test/generated/log2d1/output.csv", ",")

        val received = getMem(output_DRAM)

        Foreach(I32(iterations) by I32(1)) {
          i =>
            val iter = i + offset
            val gold = golden(iter)
            val rec = received(i)
            val diff = abs(gold - rec)
            assert(diff < 0.001.toUnchecked[T], r"$diff > 0.001 at iteration $iter. Expected: $gold, received: $rec")
        }
    }
    assert(Bit(true), "Compiles and runs")

    println(r"Config: $repeat_iterations, $run_iterations, ${config}")
  }
}

class rtl_d1l2r4c16s1_log2_coproc_main(config: LatticeConfig) extends rtl_d1l2r4c16s1_log2_coproc(16, 1, config)

class rtl_d1l2r4c16s1_log2_coproc_unrolled extends rtl_d1l2r4c16s1_log2_coproc_main(config = LatticeConfig(LatticeOptions.Unrolled))
class rtl_d1l2r4c16s1_log2_coproc_streamed1 extends rtl_d1l2r4c16s1_log2_coproc_main(config = LatticeConfig(LatticeOptions.Streamed(1)))
class rtl_d1l2r4c16s1_log2_coproc_flattened1 extends rtl_d1l2r4c16s1_log2_coproc_main(config = LatticeConfig(LatticeOptions.Flattened(1)))
class rtl_d1l2r4c16s1_log2_coproc_recursive1 extends rtl_d1l2r4c16s1_log2_coproc_main( config = LatticeConfig(LatticeOptions.Recursive(1)))

class rtl_d1l2r4c16s1_log2_coproc_streamed2 extends rtl_d1l2r4c16s1_log2_coproc_main( config = LatticeConfig(LatticeOptions.Streamed(2)))
class rtl_d1l2r4c16s1_log2_coproc_flattened2 extends rtl_d1l2r4c16s1_log2_coproc_main(config = LatticeConfig(LatticeOptions.Flattened(2)))
class rtl_d1l2r4c16s1_log2_coproc_recursive2 extends rtl_d1l2r4c16s1_log2_coproc_main(config = LatticeConfig(LatticeOptions.Recursive(2)))