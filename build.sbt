name := "mlir4spatial-libraries"

version := "0.1"

scalaVersion := "2.12.7"

val paradise_version  = "2.1.0"
/** Macro Paradise **/
resolvers += Resolver.sonatypeRepo("snapshots")
resolvers += Resolver.sonatypeRepo("releases")
addCompilerPlugin("org.scalamacros" % "paradise" % paradise_version cross CrossVersion.full)

scalacOptions ++= Seq("-explaintypes")

unmanagedSourceDirectories in Compile += baseDirectory.value / "src/mlir_libraries/"
unmanagedSourceDirectories in Compile += baseDirectory.value / "src/tensorflow_lattice/"
unmanagedSourceDirectories in Compile += baseDirectory.value / "src/spatial/"
unmanagedSourceDirectories in Compile += baseDirectory.value / "src/models/"
unmanagedSourceDirectories in Compile += baseDirectory.value / "generated/"

//resolvers += "Local Maven Repository" at "file://"+Path.userHome.absolutePath+"/.m2/repository/"

libraryDependencies ++= Seq(
  "edu.stanford.cs.dawn" %% "spatial" % "1.1-SNAPSHOT-nzhang"
)

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.5" % "test"

Test / javaSource := baseDirectory.value / "test/"

testForkedParallel in IntegrationTest := true
concurrentRestrictions in Global := Seq(Tags.limitAll(32))
