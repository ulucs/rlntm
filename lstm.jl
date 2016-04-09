using Knet, JLD

@knet function wbftwo(x,h;f=:relu,binit=Constant(0),wdims=(1,1),winit=Gaussian(0,0.1))
	w1 = par(init=winit, dims=wdims)
	w2 = par(init=winit, dims=wdims)
	b = par(init=binit, dims=(1,1))
	return f(w1*x .+ w2*h .+ b)
end

@knet function wbfone(x;f=:relu,binit=Constant(0),wdims=(1,1),winit=Gaussian(0,0.1))
	w1 = par(init=winit, dims=wdims)
	b = par(init=binit, dims=(1,1))
	return f(w1*x .+ b)
end

@knet function lstmi(x; fbias=1, lsize=100, insize=11)
    input  = wbftwo(x,h; f=:sigm, wdims=(lsize,insize),winit=Xavier())
    forget = wbftwo(x,h; f=:sigm, wdims=(lsize,insize),winit=Xavier(), binit=Constant(fbias))
    output = wbftwo(x,h; f=:sigm, wdims=(lsize,insize),winit=Xavier())
    newmem = wbftwo(x,h; f=:tanh, wdims=(lsize,insize),winit=Xavier())
    cell = input .* newmem .+ cell .* forget
    h  = tanh(cell) .* output
    return h
end

@knet function ntm(x; lsize=100, lout=2, insize=11)
	x0 = wbfone(x; wdims=(lsize,insize))
	a = lstmi(x0; lsize=lsize, insize=lsize)
	return wbfone(a; f=:soft, wdims=(lout,lsize))
end

function runtm(f,inputstr)
	char2int = Dict('1'=>1,'2'=>2,'3'=>3,'4'=>4,'5'=>5,'6'=>6,'7'=>7,'8'=>8,'9'=>9,'0'=>10,'c'=>11,'r'=>12,'n'=>13,'s'=>14)
	int2tur = [">",",","+","."]
	out = ""
	reset!(f)
	## create the input and push forward
	for c in inputstr
		imp = zeros(Float32,length(char2int))
		imp[char2int[c]] = 1
		res = forw(f,imp)
		ind = findmax(res)[2]
		out = out*int2tur[ind]
	end
	return ntmpreter(inputstr,out)
end

function ntmpreter(strin,strout)
	tur2int = Dict('>'=>1,','=>2,'+'=>3,'.'=>4)
	funcdict = Dict(
		'>' => x -> 0,
		',' => x -> push!(out,x),
		'+' => x -> push!(mem,x),
		'.' => x -> push!(out,pop!(mem))
		)
	output = ""
	n = 0
	out = Any[]
	mem = Any[]
	for c in strout
		n += 1
		funcdict[c](strin[n])
	end
	for it in out
		output = string(output,it)
	end
	return output
end

function traintm(f, data; loss=softloss, nforw=10, gclip=3.0)
    reset!(f)
    ystack = Any[]
    T = length(data[1])
    for t = 1:T
        x = data[1][t]
        y = data[2][t]
        sforw(f, x; dropout=true)
        push!(ystack, y)
        if (t % nforw == 0 || t == T)
            while !isempty(ystack)
                ygold = pop!(ystack)
                sback(f, ygold, loss)
            end
            update!(f; gclip=gclip)
            reset!(f; keepstate=true)
        end
    end	
end

function initcopy()
	batchsize = 68
	batchnum = 100
	testnum = 10
	##initial = load("trainingbatches.jld")
	## data0 = initial["traindata"]
	## datat0 = initial["testdata"]
	##char2int = initial["chardict"]
	char2int = Dict('1'=>1,'2'=>2,'3'=>3,'4'=>4,'5'=>5,'6'=>6,'7'=>7,'8'=>8,'9'=>9,'0'=>10,'c'=>11)
	tur2int = Dict('>'=>1,','=>2)
	##tur2int = initial["turdict"]
	## retrieve data from txt instead
	## data = (Any[],Any[])
	## data0 = (replace(readall("traindata.txt"),"\r\n",""),replace(readall("trainout.txt"),"\r\n",""))
	
	## better idea: create random training date to avoid overfitting
	## for i=1:batchnum
	## 	d = zeros(Float32, length(char2int), batchsize)
	## 	for j=1:batchsize
	## 		d[char2int[data0[1][i+(j-1)*batchnum]],j] = 1
	## 	end
	## 	push!(data[1],d)
	## end
	## for i=1:batchnum
	## 	d = zeros(Float32, length(tur2int), batchsize)
	## 	for j=1:batchsize
	## 		d[tur2int[data0[2][i+(j-1)*batchnum]],j] = 1
	## 	end
	## 	push!(data[2],d)
	## end

	data = (Any[],Any[])
	for i=1:batchnum
		d = zeros(Float32, 11, batchsize)
		d1 = zeros(Float32, length(tur2int), batchsize)
		rnum = rand(1:11,batchsize*batchnum)
		for j=1:batchsize
			# generate a character here
			d[rnum[i+(j-1)*batchnum],j] = 1
			##if j>1 && rnum[i+(j-2)*batchnum] != 11
			##	rout = rnum[i+(j-1)*batchnum] < 11 ? 2 : 1
			##else
			##	rout = 1
			##end
			rout = rnum[i+(j-1)*batchnum] < 11 ? 2 : 1
			d1[rout,j] = 1
		end
		push!(data[1],d)
		push!(data[2],d1)
	end

	## currently importing directly from txt files
	testdata= (Any[],Any[])
	datat0 = (replace(readall("testx.txt"),"\r\n",""),replace(readall("testout.txt"),"\r\n",""))
	for i=1:testnum
		d = zeros(Float32, 11, batchsize)
		for j=1:batchsize
			d[char2int[datat0[1][i+(j-1)*testnum]],j] = 1
		end
		push!(testdata[1],d)
	end
	for i=1:testnum
		d = zeros(Float32, length(tur2int), batchsize)
		for j=1:batchsize
			d[tur2int[datat0[2][i+(j-1)*testnum]],j] = 1
		end
		push!(testdata[2],d)
	end
	return data, testdata
end

function initcopyskip()
	batchsize = 60
	batchnum = 9
	char2int = Dict('1'=>1,'2'=>2,'3'=>3,'4'=>4,'5'=>5,'6'=>6,'7'=>7,'8'=>8,'9'=>9,'0'=>10,'c'=>11,'r'=>12,'n'=>13)
	tur2int = Dict('>'=>1,','=>2,'+'=>3,'.'=>4)

	data= (Any[],Any[])
	data0 = (replace(readall("copyin.txt"),"\r\n",""),replace(readall("skipout.txt"),"\r\n",""))
	for i=1:batchnum
		d = zeros(Float32, length(char2int), batchsize)
		for j=1:batchsize
			d[char2int[data0[1][i+(j-1)*batchnum]],j] = 1
		end
		push!(data[1],d)
	end
	for i=1:batchnum
		d = zeros(Float32, length(tur2int), batchsize)
		for j=1:batchsize
			d[tur2int[data0[2][i+(j-1)*batchnum]],j] = 1
		end
		push!(data[2],d)
	end

	## currently importing directly from txt files
	testdata= (Any[],Any[])
	## datat0 = (replace(readall("testx.txt"),"\r\n",""),replace(readall("testskipout.txt"),"\r\n",""))
	## for i=1:testnum
	## 	d = zeros(Float32, 11, batchsize)
	## 	for j=1:batchsize
	## 		d[char2int[datat0[1][i+(j-1)*testnum]],j] = 1
	## 	end
	## 	push!(testdata[1],d)
	## end
	## for i=1:testnum
	## 	d = zeros(Float32, length(tur2int), batchsize)
	## 	for j=1:batchsize
	## 		d[tur2int[datat0[2][i+(j-1)*testnum]],j] = 1
	## 	end
	## 	push!(testdata[2],d)
	## end
	return data, testdata
end

function initreverse()
	batchsize = 11
	batchnum = 60
	testnum = 10
	char2int = Dict('1'=>1,'2'=>2,'3'=>3,'4'=>4,'5'=>5,'6'=>6,'7'=>7,'8'=>8,'9'=>9,'0'=>10,'c'=>11,'r'=>12,'n'=>13)
	tur2int = Dict('>'=>1,','=>2,'+'=>3,'.'=>4)

	data= (Any[],Any[])
	data0 = (replace(readall("reversein.txt"),"\r\n",""),replace(readall("reverseout.txt"),"\r\n",""))
	for i=1:batchnum
		d = zeros(Float32, length(char2int), batchsize)
		for j=1:batchsize
			d[char2int[data0[1][i+(j-1)*batchnum]],j] = 1
		end
		push!(data[1],d)
	end
	for i=1:batchnum
		d = zeros(Float32, length(tur2int), batchsize)
		for j=1:batchsize
			d[tur2int[data0[2][i+(j-1)*batchnum]],j] = 1
		end
		push!(data[2],d)
	end

	testdata= (Any[],Any[])
	return data, testdata
end

function initcopyrev()
	batchsize = 120
	batchnum = 11
	testnum = 10
	char2int = Dict('1'=>1,'2'=>2,'3'=>3,'4'=>4,'5'=>5,'6'=>6,'7'=>7,'8'=>8,'9'=>9,'0'=>10,'c'=>11,'r'=>12,'n'=>13)
	tur2int = Dict('>'=>1,','=>2,'+'=>3,'.'=>4)

	data= (Any[],Any[])
	data0 = (replace(readall("revcopyin.txt"),"\r\n",""),replace(readall("revcopyout.txt"),"\r\n",""))
	for i=1:batchnum
		d = zeros(Float32, length(char2int), batchsize)
		for j=1:batchsize
			d[char2int[data0[1][i+(j-1)*batchnum]],j] = 1
		end
		push!(data[1],d)
	end
	for i=1:batchnum
		d = zeros(Float32, length(tur2int), batchsize)
		for j=1:batchsize
			d[tur2int[data0[2][i+(j-1)*batchnum]],j] = 1
		end
		push!(data[2],d)
	end

	testdata= (Any[],Any[])
	return data, testdata
end

function initcopyrevskip()
	batchsize = 180
	batchnum = 11
	testnum = 10
	char2int = Dict('1'=>1,'2'=>2,'3'=>3,'4'=>4,'5'=>5,'6'=>6,'7'=>7,'8'=>8,'9'=>9,'0'=>10,'c'=>11,'r'=>12,'n'=>13,'s'=>14)
	tur2int = Dict('>'=>1,','=>2,'+'=>3,'.'=>4)

	data= (Any[],Any[])
	data0 = (replace(readall("revcopyskipin.txt"),"\r\n",""),replace(readall("revcopyskipout.txt"),"\r\n",""))
	for i=1:batchnum
		d = zeros(Float32, length(char2int), batchsize)
		for j=1:batchsize
			d[char2int[data0[1][i+(j-1)*batchnum]],j] = 1
		end
		push!(data[1],d)
	end
	for i=1:batchnum
		d = zeros(Float32, length(tur2int), batchsize)
		for j=1:batchsize
			d[tur2int[data0[2][i+(j-1)*batchnum]],j] = 1
		end
		push!(data[2],d)
	end

	testdata= (Any[],Any[])
	return data, testdata
end