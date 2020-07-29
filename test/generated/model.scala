package generated

import spatial.libdsl._
import mlir_libraries.ConversionImplicits._
@spatial class model(lattice_loops: Int, pwl_iterations: Int) extends SpatialTest {

  import spatial.dsl._

  override def compileArgs: Args = super.compileArgs and "--vv"

  type T = FixPt[TRUE, _4, _12]
  val iterations = 1000
  implicit val cfg = mlir_libraries.OptimizationConfig(lattice_loops = lattice_loops, pwl_iterations = pwl_iterations)
  def main(args: Array[String]) : Unit = {
    val dram_0 = DRAM[T](iterations, 1)
    val dram_1 = DRAM[T](iterations, 1)
    val dram_2 = DRAM[T](iterations, 1)
    val dram_3 = DRAM[T](iterations, 1)
    val dram_4 = DRAM[T](iterations, 1)
    val output_DRAM = DRAM[T](iterations)

    Accel {
      val sram_0 = SRAM[T](iterations, 1)
      val sram_1 = SRAM[T](iterations, 1)
      val sram_2 = SRAM[T](iterations, 1)
      val sram_3 = SRAM[T](iterations, 1)
      val sram_4 = SRAM[T](iterations, 1)
      Parallel {
        sram_0 load dram_0
        sram_1 load dram_1
        sram_2 load dram_2
        sram_3 load dram_3
        sram_4 load dram_4
      }

      val output_sram = SRAM[T](iterations)
      val result = model_callable.model_callable(sram_0, sram_1, sram_2, sram_3, sram_4)
      Pipe.Foreach(iterations by 1) { i =>
        output_sram(i) = result(i, 0)()
      }

      output_DRAM store output_sram
    }
    println(getMem(output_DRAM))
    assert(Bit(true), "Compiles and runs")
  }
}

@spatial class model_lazy(lattice_loops: Int, pwl_iterations: Int) extends SpatialTest {

  import spatial.dsl._

  override def compileArgs: Args = super.compileArgs and "--vv"

  type T = FixPt[TRUE, _4, _12]
  val iterations = 1000
  implicit val cfg = mlir_libraries.OptimizationConfig(lattice_loops = lattice_loops, pwl_iterations = pwl_iterations)
  def main(args: Array[String]) : Unit = {
    val dram_0 = DRAM[T](iterations, 1)
    val dram_1 = DRAM[T](iterations, 1)
    val dram_2 = DRAM[T](iterations, 1)
    val dram_3 = DRAM[T](iterations, 1)
    val dram_4 = DRAM[T](iterations, 1)
    val output_DRAM = DRAM[T](iterations)

    Accel {
      val sram_0 = SRAM[T](iterations, 1)
      val sram_1 = SRAM[T](iterations, 1)
      val sram_2 = SRAM[T](iterations, 1)
      val sram_3 = SRAM[T](iterations, 1)
      val sram_4 = SRAM[T](iterations, 1)
      Parallel {
        sram_0 load dram_0
        sram_1 load dram_1
        sram_2 load dram_2
        sram_3 load dram_3
        sram_4 load dram_4
      }

      val output_sram = SRAM[T](iterations)
      val result = model_callable.model_callable_lazy(sram_0, sram_1, sram_2, sram_3, sram_4)
      Pipe.Foreach(iterations by 1) { i =>
        output_sram(i) = result(i, 0)()
      }

      output_DRAM store output_sram
    }
    println(getMem(output_DRAM))
    assert(Bit(true), "Compiles and runs")
  }
}

class model_funroll_lazy extends model_lazy(0, 1)

class model_funroll extends model(0, 1)

