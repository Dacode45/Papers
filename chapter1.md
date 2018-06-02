# Learn Physics by Programming in Haskell

## Newtonian Mechanics



```haskell
data Vec = Vec { xComp :: Double
                ,yComp :: Double
                ,zComp :: Double }
```

| Function | Description | Type |
| :--- | :--- | :--- |
| + | vector addition  | Vec -&gt; Vec -&gt; Vec |
| -  | vector subtraction  | Vec -&gt; Vec -&gt; Vec |
| \* | scalar multipication | Double -&gt; Vec -&gt; Vec |
| \* | Scalar multiplciation | Vec -&gt; Double -&gt; Vec |
| ^/ | Scalar division | Vec -&gt; Double -&gt; Vec |
| . | dot product | Vec -&gt; Vec -&gt; Double |
| &gt;&lt; | cross product | Vec -&gt; Vec -&gt; Vec |
| magnitude | magnitude | Vec -&gt; Double |
| zeroV  | zero vector | Vec |
| iHat | unit vector | Vec |
| negateV | vector negation | Vec -&gt; Vec |
| vec | vector construction | Double -&gt; Double -&gt; Double -&gt; Vec |
| xComp | vector computent | vec -&gt; double |
| sumV  | vector sum | \[Vec\] -&gt; Vec |



The above is for functions for working exclusivley with vectos. We want to be able to write code that can work with numbers or vectors. 



![](/assets/Screenshot from 2018-06-02 14-00-50.png)



## Single Particle Mechanics

We can describe particles by their position and their velocity. We can include the current time as well

```haskell
type Time = Double
type Displacement = Vec
type Velocity = Vec
type State = (Time, Displacement, Velocity)

```

State changes should be an update function taking a time and returning a new state. Double -&gt; State -&gt; State

We start with the classic. The acceleration function

`type AccelerationFunction = State -> Vec`

This gives us a system of first-order defferntial equations; the rate of change of displacemnt is velocity, and the rate of change of velocity is the accelration.

We can use the [Euler method](#euler-method) to solve the differential equatoin.

```haskell
eulerStep :: AccelerationFunction -> Double -> State -> State
eulerStep a dt (t,r,v) = (t',r',v')
    where
        t' = t + dt
        r' = r ^+^ v ^* dt
        v' = v ^+^ a(t,r,v) ^* dt
```

To define any particular one-partical problem we just specify the appropriate acceleration function.

Here's an sattelite example
```haskell
satellite :: AccelerationFunction
satellite (t,r,v) = 6.67e-11 * 5.98e24 / magnitude r ^ 2 *^ u
    where
        u = negateV r ^ /  magnitude r
```
Annother is the problem of a damped, driven, harmonic oscillator.
```
dampedDrivenOsc :: Double -- damping constant
                -> Double -- drive amplitude
                -> Double -- drive frequency
                -> AccelerationFunction
dampedDrivenOsc beta driveAmp omega (t,r,v)
    = (forceDamp ^+^ forceDrvie ^+^ forceSpring) ^/ mass
        where
            forceDamp = (-beta) *^ v
            forceDrive = driveAmp * cos (omega * t) *^ iHat
            forceSpring = (-k) *^ r
            mass = 1
            k = 1
```

Using haskell we can generate an infinite list of state changes. 
```
solution :: AccelerationFunctoin -> Double -> State -> [State]
solution a dt = iterate (eulerCromerStep a dt)

states :: [State]
states = solution (dampedDrivenOsc 0 1 0.7) 0.01 (0, vec 1 0 0, vec 0 0 0)
```

We can convert that infinite list of states to plot x vs t.

txPairs :: [State] -> [(Double, Double)]
txPairs sts = [(t, xComp r) | (t,r,v) <- sts]


# Table of Contents

1. <a href="euler-method">Euler Method</a> Method to solve differential equations. Approximate a curve using its derivative by stepping through it. The step size determines the resolution.



