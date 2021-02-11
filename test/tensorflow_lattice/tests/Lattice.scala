package tensorflow_lattice.tests

import spatial.dsl._
import mlir_libraries.types.TypeImplicits._
import mlir_libraries.{CoprocConfig, CoprocOptions, CoprocessorScope, LatticeConfig, LatticeOptions, Tensor => MLTensor}

object LatticeTest {
  val iterations = 20
  val repeats = 100
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

  val golden = scala.Array(0.2088466 ,
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
}

@spatial class LatticeTest(config: mlir_libraries.LatticeConfig) extends SpatialTest {

//  override def compileArgs = "--max_cycles=2000"

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
      CoprocessorScope {
        cps =>
          implicit val c = cps
          tensorflow_lattice.tfl.Lattice(tp = "hypercube",
            shape = MLTensor(values = scala.Array(2, 2, 2, 2, 2), shape = scala.Array(5)),
            units = 1,
            lattice_kernel = LatticeTest.lattice_kernel, config = config)(input_sram)
      } {
        case (scope, lattice) =>
          val interface = lattice.getInterface
          Foreach(LatticeTest.repeats by 1, iterations by 1) {
            case (r, i) => interface.enq(Seq(i, I32(0)), Set(Bit(true)))
          }

          Sequential {
            Foreach(LatticeTest.repeats by 1, iterations by 1) {
              case (r, i) => output_sram(i) = interface.deq(Seq(i, I32(0)), Set(Bit(true)))
            }
            scope.kill()
          }
      }
      output_DRAM store output_sram
    }
    val golden = Array[T]((LatticeTest.golden map { argon.uconst[T](_) }):_*)
    val output = getMem(output_DRAM)
    printArray(output, "Output: ")
    (0 until iterations) foreach {
      i =>
        val diff = abs(golden(i) - output(i))
        assert(diff < 1e-3, r"Error at $i: Expected (${golden(i)}), got (${output(i)}) < 1e-3")
    }
  }
}

class LatticeTestUnrolled extends LatticeTest(config = LatticeConfig(LatticeOptions.Unrolled))

class LatticeTestStreamL1 extends LatticeTest(config = LatticeConfig(LatticeOptions.Streamed(1)))
class LatticeTestStreamL2 extends LatticeTest(config = LatticeConfig(LatticeOptions.Streamed(2)))
class LatticeTestStreamL3 extends LatticeTest(config = LatticeConfig(LatticeOptions.Streamed(3)))
class LatticeTestStreamL4 extends LatticeTest(config = LatticeConfig(LatticeOptions.Streamed(4)))

class LatticeTestFlattenedL1 extends LatticeTest(config = LatticeConfig(LatticeOptions.Flattened(1)))
class LatticeTestFlattenedL2 extends LatticeTest(config = LatticeConfig(LatticeOptions.Flattened(2)))
class LatticeTestFlattenedL3 extends LatticeTest(config = LatticeConfig(LatticeOptions.Flattened(3)))
class LatticeTestFlattenedL4 extends LatticeTest(config = LatticeConfig(LatticeOptions.Flattened(4)))

class LatticeTestRecursiveL1 extends LatticeTest(config = LatticeConfig(LatticeOptions.Recursive(1)))
class LatticeTestRecursiveL2 extends LatticeTest(config = LatticeConfig(LatticeOptions.Recursive(2)))
class LatticeTestRecursiveL3 extends LatticeTest(config = LatticeConfig(LatticeOptions.Recursive(3)))
class LatticeTestRecursiveL4 extends LatticeTest(config = LatticeConfig(LatticeOptions.Recursive(4)))
