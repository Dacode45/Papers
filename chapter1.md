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

# Electromagnetic Theory

## Electric field produced by continuous charge distribution

![](/assets/Screenshot from 2018-06-02 22-48-57.png)

We can define a curve as the following
```haskell
data Curve = Curve {
    curveFunc :: Double -> Position
    , startingCurveParam :: Double
    , endingCurveParam :: Double
}
-- Circular loop
circularLoop :: Double -> Curve
circularLoop radius
    = Curve (\t -> cart (radius * cost) (radius * sin t)) 0 (2 * pi)
    
line :: Double -> Curve
line l = Curve (\t -> cart 0 0 t) (-1 / 2) ( 1 / 2 )
```

Now that we have curves we need to integrate over them. Integrals can have scalar or vector integrands. To calculate the electric field we need to use a vector integrand to calculate an electric potential we can use a scalar integral.

```haskell
type ScalarField = Position -> Double
type VectorField = Position -> Vec
type Field v = Position -> v
```

A scalar field assigns a scalar to each position in space. A vector field assigns a vector to each position in space. We can refer to fields of any type with `Field v`

Let's get to the integration.

```haskell
simpleLineIntegral
    :: (InnerSpace v, Scalar v - Double)
        => Int -- ^ number of intervals
    -> Field v -- ^ scalar or vector field
    -> Curve -- ^ curve to integrate over
    -> v -- ^ scalar or vector result
```

The integrator works by chopping the curve into a number of intervals, evaluating hte field on each interval, multiplying by the lenght and summing

We can use this to calculate the electic field of a one-dimentional charge distribution

```haskell
eFieldFromLineCharge
    :: ScalarField -- ^ linear charge density lambda
    -> Curve       -- ^ geometry of the line charge
    -> VectorField -- ^ electric field (in V/m)
    
eFieldFromLineCharge lambda c r
    = k *^ simpleLineIntegral 1000 integrand c
        where
            k = 9e9 -- 1 / (4 * pi * epsilon0)
            integrand r' = lambda r' *^ d ^/ magnitude d ** 3
                where 
                    d = displacement r' r
```



# Table of Contents

1. <a href="euler-method">Euler Method</a> Method to solve differential equations. Approximate a curve using its derivative by stepping through it. The step size determines the resolution.



