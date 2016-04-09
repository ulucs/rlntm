include("lstm.jl")

(data, testdata) = initcopyrev()
f = compile(:ntm;lsize=13,insize=13,lout=4)
setp(f, lr=0.1)
clip = 1
softer = 20
n = 0
batchsize = 11
while true
	n += 1

##for n=1:95
	traintm(f,data;loss=softloss,gclip=1.0/n)

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

	## toterrs = 0
	## softers = 0
	## for i=1:68
	## 	yg = testdata[2][1]
	## 	yt = forw(f,testdata[1][1])
	## 	toterrs += zeroone(yg,yt)
	## 	softers += softloss(yg,yt)
	## end
	## reset!(f)
	## softers = softers/68
	## toterrs = toterrs/68
	## @printf("Epoch: %i\nCurrent training accuracy: %3d%%; softloss: %5.7f\nCurrent test accuracy:     %3d%%; softloss: %5.7f\n",n,(1-toterr)*100,softer,(1-toterrs)*100,softers)

	@printf("Epoch: %i\nCurrent training accuracy: %3d%%; softloss: %5.7f\n",n,(1-toterr)*100,softer)

	if toterr == 0
		println("Convergence complete!")
	 	break
	end
	## clip = clip * (softer > softm1 ? 1 : 0.99)
end