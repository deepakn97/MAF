def solution():
    """
    There are one hundred tickets to be sold for a volleyball game. Andrea sold twice as many tickets as Jude while Sandra sold 4 more than half the number of tickets Jude sold. Andrea's brother bought 50 tickets for a basketball game. If Jude sold 16 tickets, how many tickets need to be sold?
    """
    total_tickets = 100
    jude_tickets = 16
    andrea_tickets = jude_tickets * 2
    sandra_tickets = (jude_tickets + 4) // 2
    brother_tickets = 50
    tickets_sold = andrea_tickets + sandra_tickets
    tickets_remaining = total
