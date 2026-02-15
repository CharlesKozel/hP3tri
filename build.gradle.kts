plugins {
    kotlin("jvm") version "2.1.10" apply false
    kotlin("plugin.serialization") version "2.1.10" apply false
//    idea
}

allprojects {
    repositories {
        mavenCentral()
    }
}

//idea {
//    module {
//        // Must use assignment — .add()/.addAll() silently fail (gradle/gradle#8749)
//        excludeDirs = excludeDirs +
//            file("python") +
//            file("web/node_modules") +
//            file("web/dist") +
//            file(".venv")
//    }
//}
