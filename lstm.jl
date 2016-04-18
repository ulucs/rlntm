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
	return ntmpreter(inputstr,ntmactions(f,inputstr))
end

function ntmactions(f,inputstr)
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
	return out
end

function ntmpreter(strin,strout)
	tur2int = Dict('>'=>1,','=>2,'+'=>3,'.'=>4)
	funcdict = Dict(
		'>' => x -> 0,
		',' => x -> push!(out,x),
		'+' => x -> push!(mem,x),
		'.' => x -> length(mem)>0 ? push!(out,pop!(mem)) : 0
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

function reinforcegrad(strex,strout,ypred,ydata;a=(p,ro)->ro*p*(1-p),b=0)
	## this is assuming bernoulli distribution
	## support for different a values comes later
	ygrad = zeros(Float32,size(ypred))
	numberex = parse(Float32,strex)
	numberout = 0
	try; numberout = parse(Float32,strout); end
	r = numberex == numberout ? 1 : -1
	y = zeros(Float32,size(ypred))
	ro = (1+tanh(randn(size(ypred))))
	## construct the 0-1 array for y
	for i=1:size(y,2)
		_,m = findmax(ydata[:,i])
		y[:,i] = zeros(size(y[:,i]))
		y[m,i] = 1
	end
	## calculate the ygrad
	ygrad = ro.*r.*(y.-ypred)
	return ygrad
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
	## retrieve data trainingdata/from txt instead
	## data = (Any[],Any[])
	## data0 = (replace(readall("trainingdata/traindata.txt"),"\r\n",""),replace(readall("trainingdata/trainout.txt"),"\r\n",""))
	
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

	## currently importing directly trainingdata/from txt files
	testdata= (Any[],Any[])
	datat0 = (replace(readall("trainingdata/testx.txt"),"\r\n",""),replace(readall("trainingdata/testout.txt"),"\r\n",""))
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
	data0 = (replace(readall("trainingdata/copyin.txt"),"\r\n",""),replace(readall("trainingdata/skipout.txt"),"\r\n",""))
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

	## currently importing directly trainingdata/from txt files
	testdata= (Any[],Any[])
	## datat0 = (replace(readall("trainingdata/testx.txt"),"\r\n",""),replace(readall("trainingdata/testskipout.txt"),"\r\n",""))
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
	data0 = (replace(readall("trainingdata/reversein.txt"),"\r\n",""),replace(readall("trainingdata/reverseout.txt"),"\r\n",""))
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
	data0 = (replace(readall("trainingdata/revcopyin.txt"),"\r\n",""),replace(readall("trainingdata/revcopyout.txt"),"\r\n",""))
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
	data0 = (replace(readall("trainingdata/revcopyskipin.txt"),"\r\n",""),replace(readall("trainingdata/revcopyskipout.txt"),"\r\n",""))

	instrings = split(readall("trainingdata/revcopyskipin.txt"),"\r\n")
	outstrings = split(readall("trainingdata/revcopyskipoutreinforce.txt"),"\r\n")

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
	return data, testdata, instrings, outstrings
end

function initrlshort()
	batchsize = 60
	batchnum = 5
	testnum = 10
	char2int = Dict('1'=>1,'2'=>2,'3'=>3,'4'=>4,'5'=>5,'6'=>6,'7'=>7,'8'=>8,'9'=>9,'0'=>10,'c'=>11,'r'=>12,'n'=>13,'s'=>14)
	tur2int = Dict('>'=>1,','=>2,'+'=>3,'.'=>4)

	data= (Any[],Any[])
	data0 = (replace(readall("trainingdata/shortrlin.txt"),"\r\n",""),replace(readall("trainingdata/shortrlactions.txt"),"\r\n",""))

	instrings = split(readall("trainingdata/shortrlin.txt"),"\r\n")
	outstrings = split(readall("trainingdata/shortrlout.txt"),"\r\n")

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
	return data, testdata, instrings, outstrings
end