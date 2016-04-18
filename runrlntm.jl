include("lstm.jl")

(data, _, instrings, outstrings) = initrlshort()
if !(isdefined(:persist) && persist)
	f = compile(:ntm;lsize=5,insize=14,lout=4)
end
setp(f, lr=0.001)
clip = 1
softer = 20
n = 0
batchsize = 5
batchnum = 60
while true && n<2000
	n += 1

##for n=1:95
	reset!(f)
	for i=1:length(instrings)
		ygrad = Any[]
		totgrad = 0

		instring = instrings[i]
		outstring = runtm(f,instring)
		outstringex = outstrings[i]

		for j=1:length(instrings[i])
			ypred = sforw(f,data[1][j][:,i])
			push!(ygrad,reinforcegrad(outstringex,outstring,ypred,data[2][j][:,i]))
		end

		for j=1:length(instrings[i])
			sback(f,pop!(ygrad))
		end

		update!(f,gclip=0.1)
		reset!(f)
	end

	reset!(f)
	softm1 = softer
	toterr = 0
	softer = 0
	for i=1:batchsize
		yg = data[2][i]
		yt = forw(f,data[1][i])
		toterr += zeroone(yg,yt)
		softer += softloss(yg,yt)
	end
	reset!(f)
	softer = softer/batchsize
	toterr = toterr/batchsize

	@printf("Epoch: %i\nCurrent training accuracy: %3d%%; softloss: %5.7f\n",n,(1-toterr)*100,softer)

	if toterr == 0
		println("Convergence complete!")
		break
	end
	## clip = clip * (softer > softm1 ? 1 : 0.99)
end