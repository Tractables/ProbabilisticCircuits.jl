function little_2var()
    v = Var(2)
    pos = compile(PlainLogicCircuit, var2lit(v))
    neg = compile(PlainLogicCircuit, -var2lit(v))
    or1 = pos | neg
    or2 = pos | neg

    v = Var(1)
    pos = compile(PlainLogicCircuit, var2lit(v))
    neg = compile(PlainLogicCircuit, -var2lit(v))
    
    and1 = pos & or1
    and2 = neg & or2
    and1 | and2
end

function little_3var()
    or1 = little_2var()
    v = Var(3)

    pos = compile(PlainLogicCircuit,  var2lit(v))
    neg = compile(PlainLogicCircuit, -var2lit(v))
    
    or2 = disjoin(children(or1))
    
    and1 = pos & or1
    and2 = neg & or2
    and1 | and2
end

function little_3var_constants()
    or1 = little_2var()
    v = Var(3)

    t = compile(PlainLogicCircuit, true)
    f = compile(PlainLogicCircuit, false)

    pos = compile(PlainLogicCircuit,  var2lit(v)) & t
    neg = compile(PlainLogicCircuit, -var2lit(v)) & f
    
    or2 = disjoin(children(or1))
    
    and1 = pos & or1
    and2 = neg & or2
    and1 | and2
end

function little_4var()
    ors = map(1:4) do v
        v = Var(v)
        pos = compile(PlainLogicCircuit, var2lit(v))
        neg = compile(PlainLogicCircuit, - var2lit(v))
        or = pos | neg
    end
    and1 = ors[1] & ors[2]
    and2 = ors[3] & ors[4]
    or = and1 | and2
end

function little_5var()
    c_4var = little_4var()
    v = Var(5)
    pos = compile(PlainLogicCircuit, var2lit(v))
    neg = compile(PlainLogicCircuit, - var2lit(v))
    or = pos | neg
    and = c_4var & or
    Plain⋁Node([and])
end

function readme_sdd()
    manager = SddMgr(7, :balanced)

    sun, rain, rainbow, cloud, snow, los_angeles, belgium = pos_literals(Sdd, manager, 7)
  
    sdd = (rainbow & sun & rain) | (-rainbow)
    sdd &= (-los_angeles | -belgium) 
    sdd &= (los_angeles ⇒ sun) ∧ (belgium ⇒ cloud)
    sdd &= (¬(rain ∨ snow) ⇐ ¬cloud)
    sdd
end