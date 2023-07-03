# litt_disparity

### To Dos:
- [ ] create figure

Per Aden:

"Hey Andreas,

Received the code from the Emory group. I'm happy to adapt our data to it but am assuming you can do this faster and with more underlying knowledge.

After mulling things over, I think we should just attempt to replicate their approach in Fig. 5. You are correct in that the incidence pathologies may differ by race but given the mixed indications and numbers in our dataset already, I don't think it is worth taking the time to attempt to control for. Here is what I am thinking:

My goal is to plot LITT patients per 100,000 by zip code vs % of population that is a minority by zip code on a map and as a correlation plot (I can construct the XY plot once we have the data). We would need to get the 5-year American Community Survey estimates of minority populations (in our case, this would only consist of Hispanic or Latino of any race, Asian, or Black/African American as these were the groups in our dataset) by zip code tract area and sum these, then divide them by the total zip code population x 100 to get a % of population minority. Similarly, we would divide the LITT patients from each zip code by the total zip code population x 100000 to get this "LITT incidence" variable. They included code to do the ACS pull, we just need to get the correct columns. 

For the map size, I think we could either try the Eastern seaboard + southeast or we could do an approx 300 mi radius as this was the upper end of the SD of distance traveled for non-Hispanic White patients."
