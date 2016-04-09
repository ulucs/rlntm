include("lstm.jl")

(data, testdata) = initcopy()
f = compile(:ntm;lsize=11,insize=11,lout=2)
setp(f, lr=1.0)
n = 0
while true
	n += 1
##for n=1:95
	traintm(f,data;loss=softloss,gclip=1/n)

	toterr = 0
	softer = 0
	for i=1:10
		yg = data[2][i]
		yt = forw(f,data[1][i])
		toterr += zeroone(yg,yt)
		softer += softloss(yg,yt)
	end
	reset!(f)
	softer = softer/10
	toterr = toterr/10

	toterrs = 0
	softers = 0
	for i=1:10
		yg = testdata[2][i]
		yt = forw(f,testdata[1][i])
		toterrs += zeroone(yg,yt)
		softers += softloss(yg,yt)
	end
	reset!(f)
	softers = softers/10
	toterrs = toterrs/10
	@printf("Epoch: %i\nCurrent training accuracy: %3d%%; softloss: %5.7f\nCurrent test accuracy:     %3d%%; softloss: %5.7f\n",n,(1-toterr)*100,softer,(1-toterrs)*100,softers)

	if toterrs == 0 && toterr == 0
		println("Convergence complete!")
	 	break
	end
end