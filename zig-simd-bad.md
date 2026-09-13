---
title: "An introduction to SIMD (with Zig ezamples)"
url: "https://www.reddit.com/r/Zig/comments/1v48k7a/an_introduction_to_simd_with_zig_ezamples/"
author: "Real_Dragonfruit5048"
---

---

## Comments

> **jnordwick** · [2026-07-23](https://reddit.com/r/Zig/comments/1v48k7a/comment/oz9gboo/) · 28 points
>
> edit: I'm still banned from ziggit, or I would post it there to get more screen time in front of the rest of zig.
>
> Simd in zig is ass right now. And not likely to get better from what ive seen in the forums from core.
>
> I've done a lot of apl/j/k programming and hpc work so my opinions are colored by fitst class vector languages and workloads.
>
> Ive been saying all this shit for years but the core team is so damn dense and stubborn. Ive seen the comments they make, and they are in over their heads. The guy leading some of this claimed all this shader experience but didnt seem know about auto broadcast in the shader language he used. Pathetic.
>
> I tried to write a high performance simd math library but had issues that couldn't be worked around so I gave up.
>
> 1.  There is no auto broadcast so you are constant writing splat everywhere for constants. Every other vector system does this. At least have splat to take a scalar argument and be a noop. I had to write a splat function that did this, but it should broadcast automatically. Vector + scalar is well worn territory. Everything all the way back to apl did this.
> 2.  Select should work on scalar (essentially be a conditional) so you can keep the same code path for vector and scalar. There is a partial workaround for this and above by making a single lane vector and that work for tail vector cleanup usually but sometimes you don't want the vector context so doesn't work in every case.
> 3.  Missing critical instruction access. There is no access to rcp approx inverse, needed for fast division (simd division is mico coded as loop internaly that dispatches 2-4 lanes at a time and very slow so rcp + Newton is the usual way) I had to write my own vectorized reciprocal approximation, but there are specific instructions that are even faster.. avx512 has a lot of very useful logic, scanning, and other instructions.
> 4.  There is no access to scatter/gather vector load/store. No access to bf16 type (absolutely necessary). No native mask register type.
> 5.  Some things like @sin lower to a scalar loop and is deceiving unless you dump the asm.
> 6.  In-line asm is crippled. You cannot pass or receive vector register types in zig asm so that removes the last escape hatch you could use to solve this. You used to be able to emit llvm builtin symbols like functions that gave you almost full access to everything, but that was also removed.
>
> Zig simd is only good for light work, and doesn't appear to be pointing in the right direction. The core team is prioritizing graphics and game shaders with new vec3 types. That isn't really simd code like heavy large vector and matrix code. It is more like a struct and involved a lot of cross lane and horizontal moment (eg swizzle). There are different types programming, and making a shader language doesnt solve the same problems as hpc/ai large matmul, simd numerics, or strings.
>
> Ive offered my ability and experience on some of this but am one of the people Kelley has blocked from zig github and all groups. (He wrote a blog apology to this seemingly large group of people a few months ago but I don't think he actually unblocked anybody).
>
> > **fluidtoons** · [2026-07-23](https://reddit.com/r/Zig/comments/1v48k7a/comment/oza5jg3/) · 4 points
> >
> > If you were to recommend and intro book or something to this stuff, what would you recommend? I'm impressed and curious
> >
> > > **jnordwick** · [2026-07-23](https://reddit.com/r/Zig/comments/1v48k7a/comment/ozaazjv/) · 7 points
> > >
> > > I honestly just learned from reading agner fog's simd library, browsing stack overflow questions that looked interesting (I never thought I was say this, but I miss SO).
> > >
> > > And I then I just starting writing small pieces of code. And I would start in C where you have access to \_mm functions and types so you will learn what the hardware actually does (I think rust and maybe odin allow this too, but not sure). simd code you still need to know about how code is executed to get good performance.
> > >
> > > I used godbolt a lot for this too for its click reference for browsing code. And LLMs are excellent at giving an asm dump and asking wtf.
> > >
> > > I started with simple things like sum an array, but only use simd instructions not pseudo instructions. eg, horizontal add (sum all the values in a single vector register) is actually about 6-12 instructions: a series of folds and adds. And it is very expensive since all cross lane instructions share a single execution port (port 5 on x64) so very poor overlap. There is the misconception that every \_mm function or @reduce in zig is just like adding two vector registers. They are wildly different in cost and execution profile.
> > >
> > > I started by writing horizontal sum, then sum entire array (there you learn about how to handle the tail elements). you also learn order operation to keep cross lane traffic to a minimum (eg, to add an entire array, you don't do horizontal sums all the way down, you add vector width blocks to each other then do a single horizontal sum at the end).
> > >
> > > writing simd sin/cos is pretty easy after you know the math and approach for it, but is a good 50-100 lines where if using avx512 you learn selection masks.
> > >
> > > I wrote a bunch of trig, exponential, and linear algebra stuff first before moving on to the rest of the very large instruction set of avx512.
> > >
> > > I don't have any experience with NEON though. I just don't have access to the hardware not have I had a reason to learn it. And I'm just now picking up shader programming. Its not the same as simd though. Like a simd point is an array of x and parallel arrays of y and z. The graphics version of that is a single array of triples.
> > >
> > > > **\_r\_\_h\_** · [2026-07-23](https://reddit.com/r/Zig/comments/1v48k7a/comment/ozbwrw5/) · 4 points
> > > >
> > > > I'd echo this point. As someone who's written some NEON, I'd actually recommend "Modern Arm Assembly Language Programming" by Daniel Kusswurm; using intrinsics isn't that far off from writing the assembly yourself, so it definitely helps.
> > > >
> > > > Also, just reading through the [NEON SIMD instructions available](https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions?lang=en) is super helpful. It'll help you pick up stuff like learning that you can use 'fmaxnm/fminnm' when you want to ignore NaNs when taking upper/lower bounds of some set. I'm pretty sure that on x64, the only way to fully "ignore" NaNs is with masking (by comparing a number to itself to see if it's NaN).
> > > >
> > > > I'm a big fan of NEON's approach to SIMD; it's unfortunate that you're limited to 128 bits, but there's still a lot of cool stuff you can do.
>
> > **Real_Dragonfruit5048** · [2026-07-23](https://reddit.com/r/Zig/comments/1v48k7a/comment/ozafksd/) · 3 points
> >
> > Out of curiosity, what do you think about Rust's SIMD model? I mean, in Zig SIMD is part of the language, but in Rust it's more of an interface. Do you think it could be easier to fix what you described that way?
> >
> > Also, do you find any modern general-purpose language with good SIMD integration? Say C++ or Mojo
> >
> > > **jnordwick** · [2026-07-23](https://reddit.com/r/Zig/comments/1v48k7a/comment/ozandra/) · 3 points
> > >
> > > c++ has an upcoming simd library, but it hasnt been well received.
> > >
> > > I don't know much about rusts, I think it is similar to the \_mm C intrinsics just wrapped in a namespace/struct/trait.
> > >
> > > It is very difficult to do a general simd api especially over avx512, NEON, CUDA, etc. They are just very different and the reason you go to SIMD is to get tip of the knife performance.
> > >
> > > I mostly stick to the C intrinsics (including C++) for now since simd there is very mostly transparent
> > >
> > > > **csdt0** · [2026-07-23](https://reddit.com/r/Zig/comments/1v48k7a/comment/ozb9jy9/) · 3 points
> > > >
> > > > For what is worth, I've opened an issue for Zig that intrinsics are required even when there is a generic vector type, but after a couple years being ignored, it has been closed.
> > > >
> > > > > **jnordwick** · [2026-07-23](https://reddit.com/r/Zig/comments/1v48k7a/comment/ozbi71b/) · 3 points
> > > > >
> > > > > They won't do it. Zig has moved to value portability at the expense of performance now. That basically killed its reason to be I think.
> > > > >
> > > > > The generic vector type is just a hatch into the llvm object. Many things people think are zig are really llvm -- I think the only unique concept is comptime for a C-like language.
> > > > >
> > > > > I read through a lot of the simd and vec3/4 discussions and they are thinking of simd in terms of graphics and shaders, which are not really simd in the avx/NEON sense -- there is definitely cross over, but not really the same thing.
> > > > >
> > > > > I don't think they have enough experience in the topic to really understand the problem space.
> > >
> > > > **lekkerwafel** · [2026-07-23](https://reddit.com/r/Zig/comments/1v48k7a/comment/ozb7ah1/) · 1 points
> > > >
> > > > Now I am very curious to hear if you've seen the Go archsimd package and what you think of it?
> > > >
> > > > > **\_r\_\_h\_** · [2026-07-23](https://reddit.com/r/Zig/comments/1v48k7a/comment/ozbuo9w/) · 4 points
> > > > >
> > > > > Go's approach seems pretty solid. At the moment (or last I checked), they're rolling out arch-specific functionality; for instance, they're just doing x64 right now, so you have access to sse/avx, etc; and looks like they're working on arm64's NEON right now. I think they also mentioned developing a higher-abstraction "vector" api, that is supposed to be arch-independent; I agree with jnordwick, as it's very hard to generalize these sorts of things.
> > > > >
> > > > > Go's approach seems to be pretty similar to C/C++'s intrinsic approach, which is the best you're gonna get (aside from simd asm, or autovectorization). And I'd argue that direct access to simd in Go is far more important than for other languages, since the compiler can't auto-vectorize (or at least last time I checked).
> > > > >
> > > > > **jnordwick** · [2026-07-23](https://reddit.com/r/Zig/comments/1v48k7a/comment/ozbh7zp/) · 2 points
> > > > >
> > > > > I have not. I should check it out though. I'm working on a simd language similar to K/APL. I tried to write it in Zig, but started porting it over to C++ and have been looking through simd packages.
>
> > **levraiponce** · [2026-07-23](https://reddit.com/r/Zig/comments/1v48k7a/comment/ozaii5g/) · 2 points
> >
> > I'm writing the Dlang SIMD library, it's similar to simd-everywhere. I don't see the point in providing a "lowest common denominator" API (with a lot of new names), so for better or worse the API is the Intel x86 intrinsics. It's pretty cool because you write it once and it's optimized in other archs as well.
> >
> > > **jnordwick** · [2026-07-23](https://reddit.com/r/Zig/comments/1v48k7a/comment/ozaogwy/) · 3 points
> > >
> > > I think that is a good choice. I think there is a desire and a place for operator overloading and other well defined vector operations, but I think you need the escape hatch regardless. I think some higher-level constructs are good too (eg, vector + vector with same length vectors that handle the tail as well). I'm working on a K/APL like language so I handle all that myself (mostly by allocation choices that avoid having to deal with scalar tail clean up).
> > >
> > > The C intrinsics are pretty much the base I think. There's the pseudo instructions like hadd that kind of blur the line a little. and with avx512 you would need to make a lot of new functions that wind up recreating the \_mm intrinsics with more steps.
>
> > **Nuoji** · [2026-07-28](https://reddit.com/r/Zig/comments/1v48k7a/comment/p090eei/) · 2 points
> >
> > I have been a little surprised they don’t make the effort to make SIMD nice to work with. I mean Odin and C3 has much nicer support and it’s been there for them to learn from. I can just conclude SIMD is more of a novelty for the domains Zig primarily targets.
