include("lstm.jl")

(data, testdata) = initcopyrevskip()
f = compile(:ntm;lsize=5,insize=14,lout=4)
setp(f, lr=1.0)
clip = 1
softer = 20
n = 0
s = 0
batchsize = 11
batchnum = 180
while true
	n = n%2000
	n += 1
	s += 1

##for n=1:95
	traintm(f,data;loss=softloss,gclip=1/n)

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

	if s%100 == 0
		@printf("Epoch: %i\nCurrent training accuracy: %3d%%; softloss: %5.7f\n",s,(1-toterr)*100,softer)
	end

	#if toterr == 0
	#	println("Convergence complete!")
	# 	break
	#end
	## clip = clip * (softer > softm1 ? 1 : 0.99)
end