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



# Table of Contents

1. <a href="euler-method">Euler Method</a> Method to solve differential equations. Approximate a curve using its derivative by stepping through it. The step size determines the resolution.



