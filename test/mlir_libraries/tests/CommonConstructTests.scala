package mlir_libraries.tests

import emul.FixedPoint
import spatial.dsl._
import mlir_libraries.types.TypeImplicits._
import mlir_libraries.{CoprocessorScope, OptimizationConfig, types, Tensor => MLTensor}

object LatticeWithMaterializeTest {
  val iterations = 20
  val kernel_values = scala.Array(0.0110625 , 0.02584553, 0.07165873, 0.26998997, 0.06218195,
    0.1307857 , 0.29879928, 0.35565615, 0.07956052, 0.15393174,
    0.3418517 , 0.75207484, 0.08045971, 0.57496   , 0.8091949 ,
    0.8684181 , 0.06763732, 0.22077644, 0.32806575, 0.46256602,
    0.09771216, 0.4320222 , 0.43705177, 0.85577035, 0.10409617,
    0.6642047 , 0.77883446, 0.92640615, 0.5724287 , 0.89747846,
    0.93232393, 0.9742222)
  val lattice_kernel = MLTensor(values = kernel_values, shape = scala.Array(kernel_values.length, 1))

  val test_inputs = scala.Array(scala.Array(0.83698144, 0.26090825, 0.46094936, 0.01416074, 0.29254384),
    scala.Array(0.36440275, 0.47846574, 0.8263542 , 0.149206  , 0.86860653),
    scala.Array(0.22791185, 0.54610535, 0.82765338, 0.59834199, 0.5274269 ),
    scala.Array(0.94871351, 0.20483725, 0.23750775, 0.32579948, 0.27764412),
    scala.Array(0.5145573 , 0.90784648, 0.83125223, 0.62802987, 0.73632804),
    scala.Array(0.53892756, 0.45326166, 0.32915784, 0.31102663, 0.43569929),
    scala.Array(0.20378548, 0.77212361, 0.20490814, 0.22505568, 0.63751066),
    scala.Array(0.79915822, 0.33110532, 0.89005681, 0.55166117, 0.08782675),
    scala.Array(0.68449387, 0.36529986, 0.10428861, 0.05887833, 0.16973513),
    scala.Array(0.89355386, 0.02678991, 0.76889086, 0.00785817, 0.88853799),
    scala.Array(0.60388839, 0.08982874, 0.59381238, 0.59265132, 0.5345167 ),
    scala.Array(0.13215164, 0.35842367, 0.18304778, 0.47824492, 0.865445  ),
    scala.Array(0.56554153, 0.0433671 , 0.62682761, 0.44107008, 0.50340712),
    scala.Array(0.76795795, 0.12561144, 0.0669546 , 0.69814023, 0.4185415 ),
    scala.Array(0.06977059, 0.38072113, 0.15786527, 0.35526851, 0.91920637),
    scala.Array(0.42850198, 0.60380088, 0.00593869, 0.84973909, 0.74094795),
    scala.Array(0.12292964, 0.4360795 , 0.56017567, 0.98749103, 0.13822617),
    scala.Array(0.97293566, 0.2053581 , 0.57944337, 0.4667925 , 0.01774663),
    scala.Array(0.30471316, 0.63472427, 0.25880979, 0.43376756, 0.45860137),
    scala.Array(0.24711272, 0.15127798, 0.85100654, 0.62768202, 0.98968906)).take(iterations).flatten

  val golden = scala.Array(
    0.2088466 ,
    0.4225203 ,
    0.47889823,
    0.27273285,
    0.7385861 ,
    0.31199843,
    0.2993852 ,
    0.42058963,
    0.13882771,
    0.33848044,
    0.35988057,
    0.29189512,
    0.29085353,
    0.3199867 ,
    0.245725  ,
    0.5220107 ,
    0.42142862,
    0.30632746,
    0.34087235,
    0.4168449 ).take(iterations)

  val lattice_shape = MLTensor(values=scala.Array(2, 2, 2, 2, 2), shape=scala.Array(5))
}

@spatial class LatticeWithMaterializeTest extends SpatialTest {

  type T = spatial.dsl.FixPt[TRUE, _2, _30]
  val dimensions = 5

  implicit val cfg = OptimizationConfig(lattice_loops = 4)
  def main(args: Array[String]): Unit = {
    val iterations = LatticeWithMaterializeTest.iterations
    val input_DRAM = DRAM[T](iterations, dimensions)
    setMem(input_DRAM, Array((LatticeWithMaterializeTest.test_inputs map { argon.uconst[T](_) }):_*))
    val output_DRAM = DRAM[T](I32(iterations))

    Accel {
      val input_sram = SRAM[T](iterations, dimensions)
      input_sram load input_DRAM(0 :: iterations, 0 :: dimensions)
      val output_sram = SRAM[T](iterations)

      CoprocessorScope {
        c =>
          implicit val cps = c
          tensorflow_lattice.tfl.Lattice(tp = "hypercube",
            shape = LatticeWithMaterializeTest.lattice_shape,
            units = 1,
            lattice_kernel = LatticeWithMaterializeTest.lattice_kernel)(input_sram)
      } {
        case (scope, lattice) =>
          val materialized = mlir_libraries.spatiallib.MaterializeSRAM()(lattice).getInterface
          Pipe.Foreach(iterations by 1) { i =>
            materialized.enq(Seq(i, I32(0)), Set(Bit(true)))
            output_sram(i) = materialized.deq(Seq(i, I32(0)), Set(Bit(true)))
          }

          output_DRAM store output_sram
      }
    }
    val golden = Array[T]((LatticeWithMaterializeTest.golden map { argon.uconst[T](_) }):_*)
    val outputs = getMem(output_DRAM)
    (0 until iterations) foreach {
      i =>
        val diff = abs(golden(i) - outputs(i))
        println(r"Reference: ${golden(i)}, received: ${outputs(i)}")
        assert(diff < 1e-3, r"Expected (${golden(i)}), got (${outputs(i)}) < 1e-3 at output (${i})")
    }
  }
}

@spatial class LatticeWithLazyMaterializeTest extends SpatialTest {

  type T = spatial.dsl.FixPt[TRUE, _2, _30]
  val dimensions = 5

  implicit val cfg = OptimizationConfig(lattice_loops = 4)
  def main(args: Array[String]): Unit = {
    val iterations = LatticeWithMaterializeTest.iterations
    val input_DRAM = DRAM[T](iterations, dimensions)
    setMem(input_DRAM, Array((LatticeWithMaterializeTest.test_inputs map { argon.uconst[T](_) }):_*))
    val output_DRAM = DRAM[T](I32(iterations))
    println(s"BS Size: ${implicitly[argon.State].bundleStack.size}")
    Accel {
      val input_sram = SRAM[T](iterations, dimensions)
      input_sram load input_DRAM(0 :: iterations, 0 :: dimensions)
      val output_sram = SRAM[T](iterations)
      println(s"BS Size: ${implicitly[argon.State].bundleStack.size}")
      CoprocessorScope {
        scope =>
          implicit val sc = scope
          val lattice =
            tensorflow_lattice.tfl.Lattice(tp = "hypercube",
              shape = LatticeWithMaterializeTest.lattice_shape,
              units = 1,
              lattice_kernel = LatticeWithMaterializeTest.lattice_kernel)(input_sram)
          mlir_libraries.spatiallib.MaterializeCoproc()(lattice)
      } {
        case (scope, materialized) =>
          val interface = materialized.getInterface
          Stream {
            Pipe.Foreach(iterations by 1) { i =>
              interface.enq(Seq(i, I32(0)), Set(Bit(true)))
//              output_sram(i) = materialized(Seq(i, I32(0)), Set(Bit(true)))()
            }

            Pipe {
              Pipe.Foreach(iterations by 1) { i =>
                output_sram(i) = interface.deq(Seq(i, I32(0)), Set(Bit(true)))
              }
              scope.kill()
            }
          }
      }

      output_DRAM store output_sram
    }
    val golden = Array[T]((LatticeWithMaterializeTest.golden map { argon.uconst[T](_) }):_*)
    val outputs = getMem(output_DRAM)
    (0 until iterations) foreach {
      i =>
        val diff = abs(golden(i) - outputs(i))
        println(r"Reference: ${golden(i)}, received: ${outputs(i)}")
        assert(diff < 1e-3, r"Expected (${golden(i)}), got (${outputs(i)}) < 1e-3 at output (${i})")
    }
  }
}

