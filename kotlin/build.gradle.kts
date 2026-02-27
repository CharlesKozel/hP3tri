plugins {
    kotlin("jvm")
    kotlin("plugin.serialization")
    application
}

val ktor_version = "3.1.1"

application {
    applicationName = "hp3tri"
    mainClass.set("dev.ckozel.alife.hp3tri.MainKt")
    applicationDefaultJvmArgs = listOf(
        "-Djava.library.path=${project.rootDir}/.venv/lib/python3.12/site-packages/jep",
    )
}

dependencies {
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.9.0")

    implementation("io.ktor:ktor-server-core:$ktor_version")
    implementation("io.ktor:ktor-server-netty:$ktor_version")
    implementation("io.ktor:ktor-server-content-negotiation:$ktor_version")
    implementation("io.ktor:ktor-serialization-kotlinx-json:$ktor_version")
    implementation("io.ktor:ktor-server-cors:$ktor_version")

    implementation("org.slf4j:slf4j-simple:2.0.16")

    // Jep - Java Embedded Python
    implementation("black.ninia:jep:4.3.1")

    testImplementation(kotlin("test"))
    testImplementation("org.junit.jupiter:junit-jupiter:5.11.3")
}

tasks.test {
    useJUnitPlatform()
}

tasks.named<JavaExec>("run") {
    workingDir = rootDir
    val venvSitePackages = "${rootDir}/.venv/lib/python3.12/site-packages"
    environment("PYTHONPATH", venvSitePackages)
}

kotlin {
    jvmToolchain(21)
}
