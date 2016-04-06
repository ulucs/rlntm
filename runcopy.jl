include("lstm.jl")

(data, testdata) = initcopy()
f = compile(:ntm;lsize=11,insize=11,lout=2)
n = 0
while true
	n += 1
##for n=1:95
	traintm(f,data;loss=softloss)

	toterr = 0
	softer = 0
	for i=1:68
		yg = data[2][1]
		yt = forw(f,data[1][1])
		toterr += zeroone(yg,yt)
		softer += softloss(yg,yt)
	end
	reset!(f)
	softer = softer/68
	toterr = toterr/68

	toterrs = 0
	softers = 0
	for i=1:68
		yg = testdata[2][1]
		yt = forw(f,testdata[1][1])
		toterrs += zeroone(yg,yt)
		softers += softloss(yg,yt)
	end
	reset!(f)
	softers = softers/68
	toterrs = toterrs/68
	@printf("Epoch: %i\nCurrent training accuracy: %3d%%; softloss: %5.7f\nCurrent test accuracy:     %3d%%; softloss: %5.7f\n",n,(1-toterr)*100,softer,(1-toterrs)*100,softers)

	if toterrs == 0
		println("Convergence complete!")
	 	break
	end
end